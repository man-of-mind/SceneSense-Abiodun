"""Post-run bookkeeping recovery for runs_01 (no training, no environment execution).

Every run of runs_01 completed 10,000 updates, wrote its checkpoints (incl. u10000), flushed decisions.jsonl and saved
actor_u10000.pt, then raised in campaign.py while formatting RUN_RESULT.json (``Path.relative_to`` on the relative
``--out`` path; absolute ``--out`` avoids it). The training source is left unchanged so the authorization digest still
binds it. This rebuilds RUN_RESULT.json from the files on disk and verifies:
  * the u10000 bundle bytes equal its manifest sha256, its update_count is 10,000, its decision count equals the
    decisions.jsonl rows and its ledger digest equals the hash of the logged actions;
  * actor_u10000.pt equals the actor state inside the u10000 bundle, tensor by tensor;
  * metrics.jsonl has updates 1..10,000 (smoke runs: +5 duplicated continuation rows after 250 are not duplicated).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import torch

from . import authorization as A, rewards as R, trainer as TR

ROOT = Path(__file__).resolve().parents[2]


def finalize(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    out = run_dir / "RUN_RESULT.json"
    assert not out.exists(), "RUN_RESULT.json already exists"
    binding = json.loads((run_dir / "RUN_BINDING.json").read_text())
    arm, seed = binding["arm"], binding["seed_plan"]["seed"] if "seed" in binding["seed_plan"] else int(run_dir.name.split("seed")[1])
    raw = (run_dir / "checkpoints/u10000.pt").read_bytes()
    meta = json.loads((run_dir / "checkpoints/u10000.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == meta["sha256"], "bundle bytes differ from manifest"
    payload = torch.load(io.BytesIO(raw), weights_only=False)
    assert payload["update_count"] == 10_000 and payload["binding_sha256"] == binding["binding_sha256"]
    actor_path = run_dir / "actor_u10000.pt"
    actor = torch.load(actor_path, weights_only=True)
    assert actor.keys() == payload["actor"].keys() and all(torch.equal(actor[k], payload["actor"][k]) for k in actor)
    diag = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
    assert len(diag) == meta["decision_count"] == len(payload["ledger"])
    assert [[d["mode_id"], d["q_e4"]] for d in diag] == [[l[0], l[1]] for l in payload["ledger"]]
    updates = [json.loads(l)["update"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert updates == list(range(1, 10_001)), "metrics rows are not exactly updates 1..10000"
    smoke = json.loads((run_dir / "SMOKE.json").read_text()) if (run_dir / "SMOKE.json").exists() else None
    tail = diag[-4000:]
    checkpoints = sorted(int(p.stem[1:]) for p in (run_dir / "checkpoints").glob("u*.pt"))
    assert tuple(checkpoints) == TR.CHECKPOINTS
    result = dict(
        arm=arm, seed=seed, reward_sha256=R.reward_sha256(arm), updates=payload["update_count"], decisions=len(diag),
        seconds=json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])["t"],
        final_digest=meta, checkpoints=checkpoints,
        checkpoint_sha256={str(c): json.loads((run_dir / f"checkpoints/u{c:05d}.json").read_text())["sha256"] for c in checkpoints},
        actor={"path": str(actor_path.relative_to(ROOT)), "sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest()},
        smoke=smoke, sessions=meta["collector"]["session"] + 1,
        terminals={k: sum(1 for d in diag if d["terminal"] == k) for k in ("SUCCESS", "TIMEOUT", "EVALUATOR_FAULT")},
        last_4000_mode_share={m: sum(1 for d in tail if d["mode_id"] == m) / len(tail) for m in range(12)},
        last_4000_success=sum(1 for d in tail if d["terminal"] == "SUCCESS") / len(tail),
        last_4000_timely=sum(1 for d in tail if d["terminal"] != "TIMEOUT") / len(tail),
        floors={f: sum(1 for d in diag if f in d["floors"]) for f in ("D_ZERO", "E_ACCESS", "CONFIRM_SEND_AFTER_BIND", "CONFIRM_SEND_GATED")},
        recovered_by="finalize.py (post-run bookkeeping; campaign.py raised after training completed)",
        source_digest_at_training=binding["source_digest"], source_digest_now=A.source_digest())
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    for d in sys.argv[1:]:
        r = finalize(Path(d))
        print(r["arm"], r["seed"], r["terminals"], "succ %.3f timely %.3f" % (r["last_4000_success"], r["last_4000_timely"]),
              sorted(((k, round(v, 2)) for k, v in r["last_4000_mode_share"].items()), key=lambda x: -x[1])[:3],
              r["floors"], r["actor"]["sha256"][:16])
