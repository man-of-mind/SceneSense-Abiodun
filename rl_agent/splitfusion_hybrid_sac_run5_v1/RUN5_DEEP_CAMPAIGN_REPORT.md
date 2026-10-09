# Run-5 deep campaign report (2026-09-29)

**Status: COMPLETE.** Seeds 17, 29 and 43 each reached 10,000 updates (40,288
decisions) under preregistration `2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf`
(authorization token `GO_RUN5_DEEP`; launched from HEAD `a3c462f`).

This is a training-completion record only. No validation has been run, and nothing
here is a policy-performance claim.

| Item | Result |
|---|---|
| Launch checks | HEAD `a3c462f`, clean worktree, cold host, 19.06 GB free, campaign path absent |
| Process | three parallel seeds with the documented command; 962 s training plus 5 s finalization |
| Interruptions | none: each seed logged `START → COMPLETE`, with no resume or emergency bundle |
| Numerical health | all 10,000 updates per seed finite (losses, gradients, targets) |
| Checkpoints | 23 per seed (0, 100, 250, then every 500 to 10,000); every bundle passes marker, manifest and payload hash checks; `LATEST` names `checkpoint_010000` |
| Actors | final actor plus the actors at checkpoints 500/1500/2500/5000/7500/10000 cold-loaded with `weights_only` in a fresh process at seed completion, and again by `--finalize-campaign` |
| Disk | 585 MB (449 files, 611,948,444 B); all off Git |
| Completion | `CAMPAIGN_COMPLETE.json` SHA-256 `22338fcc57f5279c759228bda2c43aa142dd3f9d5c377c31b7f37a48f0944fa1` |

Registered final actors (update 10,000, by the fixed rule; no seed or checkpoint was
selected):

| Seed | Actor tensor-tree SHA-256 | `actor_state_dict.pt` SHA-256 |
|---|---|---|
| 17 | `ffd0217e87e4f2ddb19880d86f47c5fefa3178925c077c008634a1dbaba75446` | `693b1fd3ec48595348a9f4f374485ae25250d4e899674542e49ac9be57d6f2c1` |
| 29 | `01f2e0d9da05e396447edacbca2a6c3af92d48c11c6aad332c5834ff1efdeedc` | `b7d9a77a5dea77e286dd9a632ed5a8aafa2526fd315af271f2e0d92c81dc6fd7` |
| 43 | `c51b4f1af3f5f8dedc5080036ff271c55c2542ef26b40da148ccf45f1f493aff` | `f17e812defd7d14204ed9c2a140d776727ecc5cf55e0b2f5a3db466e24d2defe` |

**Provenance.** `RUN5_DEEP_CAMPAIGN_EVIDENCE.json` holds:

- `CAMPAIGN_COMPLETE.json` and all three `SEED_COMPLETE.json` contents, with their
  hashes;
- every checkpoint and final-actor manifest, keyed by seed and update;
- every `COMMITTED` marker and `LATEST` pointer;
- the run events;
- the path, size and SHA-256 of all 449 campaign files.

Running `python3 -m rl_agent.splitfusion_hybrid_sac_run5_v1.build_campaign_evidence --verify`
recomputes all of it from the untouched campaign.

**Not in Git:** `*.pt`, `event.json`, `decisions.jsonl`, `metrics.jsonl`,
`channel_state.json`, logs/PIDs and the campaign directory. They remain under
`campaign_runs/run5_three_seed_10000_v1/`, hash-bound by the evidence file.
