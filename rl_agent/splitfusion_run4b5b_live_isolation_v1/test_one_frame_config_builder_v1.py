from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from . import final_actor_gate_v2 as F
from . import one_frame_config_builder_v1 as B


class BuilderTest(unittest.TestCase):
    def test_builder_hashes_exact_files_and_keeps_all_paths_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "RUN5B_JOINT_FINAL_ACTOR_MANIFEST_V2.json"
            manifest.write_bytes(b"manifest")
            weights = root / "actor_state_dict.pt"
            weights.write_bytes(b"weights")
            value = B.build(
                run_id="one", cell_id="cell", variant=F.RUN5B_VARIANT,
                actor_manifest_path=manifest, actor_weights_path=weights,
                actor_evidence_root=root / "evidence",
                local_repository=root / "repository",
                remote_repository=Path("/srv/remote/repository"),
                local_attempt_root=root / "attempt",
                remote_attempt_root=Path("/srv/remote/attempt"),
                edge_campaign_config=Path("/srv/remote/campaign.json"),
                route_config=root / "route.json")
            self.assertEqual(value.actor_manifest_sha256,
                             hashlib.sha256(b"manifest").hexdigest())
            self.assertEqual(value.actor_weights_sha256,
                             hashlib.sha256(b"weights").hexdigest())
            self.assertEqual(value.transmitted_budget, 1)
            self.assertEqual(value.network.ue_tunnel_ip, "10.0.0.2")
            self.assertFalse(value.policy_performance_claim)


if __name__ == "__main__": unittest.main()
