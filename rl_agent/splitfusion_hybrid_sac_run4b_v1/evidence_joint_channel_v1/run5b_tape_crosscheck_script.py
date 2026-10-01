import json, sys, hashlib
from pathlib import Path
import torch; torch.set_num_threads(4)
MAIN = Path(sys.argv[1]); N = int(sys.argv[2])
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_collector as RC
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
art = MAIN / "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json"
shared = RC.build_shared_sources(art, MAIN)
out = {"run5b_source_commit": sys.argv[3], "decisions": N, "seeds": {}}
for seed in (17, 29, 43):
    c = RC.Run5BModeledCollectorV1(artifact_path=art, seed=seed, evidence_root=MAIN, shared_sources=shared)
    sched = orch.build_frozen_warmup_schedule(seed)
    tape = []
    for i in range(N):
        tape.append([int(c._mcs_current), float(c._snr_current).hex()])
        a = sched.action_at(i % len(sched))
        c.collect(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=a.mode_id, q_e4=a.q_e4,
                  source="STRATIFIED_WARMUP", warmup_q_bin_index=a.q_bin_index))
    probe = J.JointSnrMcsChannelV1(kernel=shared["snr_kernel"], design=shared["design"], seed=0)
    out["seeds"][str(seed)] = {"tape_sha256": hashlib.sha256(json.dumps(tape).encode()).hexdigest(),
                               "first5": tape[:5], "channel_binding_sha256": probe.binding_sha256,
                               "channel_seed": J.derive_seed(seed, "train-channel")}
print(json.dumps(out))
