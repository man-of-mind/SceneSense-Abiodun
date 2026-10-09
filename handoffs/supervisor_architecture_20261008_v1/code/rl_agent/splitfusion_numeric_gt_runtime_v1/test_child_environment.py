"""CPU child caps survive the retained UE's global Popen environment restore."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .process import ENV
from .test_worker_owner import _owner


class ChildEnvironmentTests(unittest.TestCase):
    def test_exec_caps_survive_parent_restore_without_changing_other_children(self):
        # Model the retained threadcap_v1 Popen shim: even an explicit env is
        # restored to the ordinary child's baseline after Client constructs it.
        real_init = subprocess.Popen.__init__
        for baseline in (None, "24"):
            with self.subTest(baseline=baseline):
                parent_before = dict(os.environ)
                restored_launches = []

                def restore_init(process, args, *positional, **keywords):
                    environment = dict(keywords.get("env", os.environ))
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                        if baseline is None:
                            environment.pop(key, None)
                        else:
                            environment[key] = baseline
                    keywords["env"] = environment
                    restored_launches.append(args)
                    return real_init(process, args, *positional, **keywords)

                with tempfile.TemporaryDirectory(prefix="numeric-exec-caps-", dir="/tmp") as root:
                    with patch.object(subprocess.Popen, "__init__", restore_init):
                        client = _owner(Path(root) / "worker", root)
                        try:
                            # Linux exposes exec-time environment here, unlike
                            # os.environ values that the worker resets later.
                            raw = Path(f"/proc/{client.process.pid}/environ").read_bytes()
                            observed = dict(item.decode().split("=", 1)
                                            for item in raw.split(b"\0") if item)
                            self.assertEqual({key: observed.get(key) for key in ENV}, ENV)
                            self.assertEqual(client.ready["torch_threads"], 1)
                            self.assertFalse(client.ready["cuda_initialized"])
                            self.assertEqual(client.call("ECHO", {"caps_at_exec": True}),
                                             {"caps_at_exec": True})
                            ordinary = subprocess.run([sys.executable, "-B", "-c",
                                "import json,os;print(json.dumps({k:os.environ.get(k) "
                                "for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')}))"],
                                env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                capture_output=True, text=True, check=True, timeout=5)
                            self.assertEqual(json.loads(ordinary.stdout),
                                {key: baseline for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")})
                            self.assertEqual(client.close(), {})
                            self.assertEqual(client.process.returncode, 0)
                            self.assertFalse(client.owner_thread_alive)
                        finally:
                            if not client.closed:
                                client.abort()
                self.assertEqual(dict(os.environ), parent_before)
                self.assertEqual(len(restored_launches), 2)


if __name__ == "__main__":
    unittest.main()
