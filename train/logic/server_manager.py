#!/usr/bin/env python3
"""
==============================================================================
GAMA SERVER MANAGER - Quan ly vong doi cua gama-server (headless WebSocket)
==============================================================================
Khoi dong / dung gama-server bang script chinh thuc:
    bash /Applications/Gama.app/Contents/headless/gama-headless.sh -socket <PORT>

gama-server lang nghe tai ws://localhost:<PORT>, nhan cac lenh JSON
(load, step, expression, reload, ...) tu Python thong qua thu vien gama-client.
==============================================================================
"""

import os
import signal
import socket
import subprocess
import time
from collections import deque

GAMA_HEADLESS = os.environ.get(
    "GAMA_HEADLESS", "/Applications/Gama.app/Contents/headless/gama-headless.sh"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")


class GamaServerError(Exception):
    pass


class GamaServerManager:
    """Quan ly subprocess cua gama-server."""

    def __init__(self, port=6868, gama_headless=GAMA_HEADLESS,
                 workdir=None, log_file=None, startup_timeout=300.0,
                 heap_mb=None):
        self.port = int(port)
        self.gama_headless = gama_headless
        self.workdir = workdir or os.path.join(RUN_DIR, "server_ws")
        self.log_file = log_file or os.path.join(RUN_DIR, "gama_server.log")
        self.startup_timeout = float(startup_timeout)
        self.heap_mb = int(heap_mb if heap_mb is not None else os.environ.get("GAMA_HEAP_MB", "2048"))
        if self.heap_mb < 512:
            raise ValueError("GAMA heap must be at least 512 MB")
        self.process = None
        self._log_fh = None
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    # ------------------------------------------------------------------ utils
    def is_port_open(self, timeout=1.0):
        """Kiem tra cong WebSocket co ai dang lang nghe khong."""
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def _tail_log(self, n=25):
        try:
            with open(self.log_file, "r", errors="replace") as f:
                return "".join(deque(f, maxlen=n))
        except OSError:
            return "(khong doc duoc log)"

    # ----------------------------------------------------------------- start
    def start(self, reuse_running=True):
        """
        Khoi dong gama-server. Neu cong da mo va reuse_running=True thi
        gia su server van chay (khong khoi dong lai).
        Tra ve True neu da co server san sang.
        """
        if self.is_port_open():
            if reuse_running:
                print(f"[GamaServer] Cong {self.port} da mo -> dung lai server hien co.")
                return True
            raise GamaServerError(f"Cong {self.port} da bi chiem boi tien trinh khac.")

        os.makedirs(self.workdir, exist_ok=True)
        self._log_fh = open(self.log_file, "w")
        configuration = os.path.join(self.workdir, "configuration")
        workspace = os.path.join(self.workdir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        cmd = ["bash", self.gama_headless, "-m", f"{self.heap_mb}m", "-socket", str(self.port),
               "-configuration", configuration, "-data", workspace]
        server_env = {
            **os.environ,
            # GAMA's launcher appends its own -Xmx16G after -m. JVM applies
            # _JAVA_OPTIONS last, keeping the requested per-worker cap real.
            "_JAVA_OPTIONS": os.environ.get("_JAVA_OPTIONS", "") +
                             f" -Xms256m -Xmx{self.heap_mb}m -XX:-AlwaysPreTouch",
        }
        print(f"[GamaServer] Dang khoi dong: {' '.join(cmd)}  (workdir={self.workdir})")
        print(f"[GamaServer] Log: {self.log_file}  (heap={self.heap_mb}MB, AlwaysPreTouch=off)")
        self.process = subprocess.Popen(
            cmd, cwd=self.workdir, stdout=self._log_fh, stderr=subprocess.STDOUT, start_new_session=True,
            env=server_env,
        )

        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            if self.process.poll() is not None:
                raise GamaServerError(
                    "gama-server ket thuc som voi code "
                    f"{self.process.returncode}. Log cuoi:\n{self._tail_log()}"
                )
            if self.is_port_open():
                print(f"[GamaServer] San sang tai ws://localhost:{self.port} "
                      f"(sau {time.time() - t0:.0f}s).")
                return True
            time.sleep(2.0)

        self.stop()
        raise GamaServerError(
            f"Timeout {self.startup_timeout}s cho gama-server mo cong {self.port}. "
            f"Log cuoi:\n{self._tail_log()}"
        )

    # ------------------------------------------------------------------ stop
    def stop(self, force=False):
        """Dung gama-server (terminate, roi kill neu can)."""
        if self.process is not None:
            # gama-headless.sh launches a JVM child. Terminating only bash leaves
            # that JVM listening with cached GAML. Signal our isolated group.
            try:
                os.killpg(self.process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                # macOS may report EPERM for a vanished process group. Only
                # accept that case after checking both launcher and listener.
                if self.process.poll() is None or self.is_port_open(timeout=.1):
                    raise
            deadline = time.monotonic() + (2 if force else 15)
            while time.monotonic() < deadline:
                # The shell can exit before the JVM finishes its shutdown
                # hooks. A process-group probe is not proof the port is free.
                if self.process.poll() is not None and not self.is_port_open(timeout=.1):
                    break
                time.sleep(0.1)
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=10)
            deadline = time.monotonic() + 10
            while self.is_port_open(timeout=.1):
                if time.monotonic() >= deadline:
                    raise GamaServerError(f"Cong {self.port} van mo sau khi dung server da tao.")
                time.sleep(.1)
            print("[GamaServer] Da dung gama-server.")
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        self.process = None

    # ----------------------------------------------------------- context mgr
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
