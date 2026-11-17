"""Host-only guard tests; no Torch, NPU API or benchmark invocation."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from benchmarks.mha.resource_guard import cpu_environment, process_tree, stop_group


class ResourceGuardTests(unittest.TestCase):
    def test_cpu_threads_override_inherited_unbounded_values(self):
        with patch.dict(
            os.environ, {"FGW_CPU_THREADS": "4", "OMP_NUM_THREADS": "256", "MKL_NUM_THREADS": "256"}
        ):
            self.assertEqual(cpu_environment(), 4)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "4")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "4")
        with patch.dict(os.environ, {"FGW_CPU_THREADS": "256"}):
            with self.assertRaises(ValueError):
                cpu_environment()

    @unittest.skipUnless(os.name == "posix" and os.path.isdir("/proc"), "Linux process groups required")
    def test_stop_only_owned_worker_group(self):
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        try:
            tree = process_tree(worker.pid)
            self.assertIn(worker.pid, tree["pids"])
            self.assertNotIn(unrelated.pid, tree["pids"])
            stop_group(worker)
            self.assertIsNotNone(worker.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for process in (worker, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
