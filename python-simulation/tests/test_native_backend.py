import cProfile
from importlib.machinery import EXTENSION_SUFFIXES
import pickle
import pstats
import unittest

from python_simulation import python_sim


@unittest.skipUnless(python_sim.__file__.endswith(tuple(EXTENSION_SUFFIXES)),
                     "optional native extensions are not active")
class NativeBackendTests(unittest.TestCase):
    def test_profile_and_worker_checkpoint(self):
        sim = python_sim.LeVanLuongSimulator(seed=1)
        profile = cProfile.Profile()
        profile.runcall(sim.step, 75)
        calls = {key[2]: value[1] for key, value in pstats.Stats(profile).stats.items()}
        self.assertEqual(calls["_move"], 75)
        self.assertGreater(calls["_neighbors"], 0)

        resumed = pickle.loads(pickle.dumps(sim))
        self.assertEqual(sim.step(50), resumed.step(50))
