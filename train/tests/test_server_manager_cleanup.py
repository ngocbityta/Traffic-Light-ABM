"""Regression: stopping the shell launcher must also stop its listener child."""
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train.logic.server_manager import GamaServerManager


@unittest.skipUnless(os.name == 'posix', 'GAMA shell launcher uses POSIX process groups')
class ServerCleanupTest(unittest.TestCase):
    def test_stop_releases_child_port(self):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', 0))
                probe.listen(1)
                port = probe.getsockname()[1]
                with socket.create_connection(('127.0.0.1', port), timeout=0.2):
                    pass
            except (PermissionError, OSError) as error:
                self.skipTest(f'local socket loopback is unavailable: {error}')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'listener.py').write_text(
                'import socket,sys,time,signal\n'
                'def delayed_exit(*args):\n time.sleep(1);sys.exit(0)\n'
                'signal.signal(signal.SIGTERM,delayed_exit)\n'
                's=socket.socket();s.bind(("127.0.0.1",int(sys.argv[1])));'
                's.listen();time.sleep(120)\n')
            launcher = root / 'headless.sh'
            launcher.write_text('"$TEST_PYTHON" listener.py "$2" &\nwait\n')
            manager = GamaServerManager(port=port, gama_headless=str(launcher), workdir=tmp,
                                        log_file=str(root / 'server.log'), startup_timeout=10)
            with patch.dict(os.environ, {'TEST_PYTHON': sys.executable}):
                try:
                    manager.start(reuse_running=False)
                    self.assertTrue(manager.is_port_open())
                finally:
                    manager.stop()
            self.assertFalse(manager.is_port_open(), 'JVM-equivalent child still owns the port')


if __name__ == '__main__':
    unittest.main()
