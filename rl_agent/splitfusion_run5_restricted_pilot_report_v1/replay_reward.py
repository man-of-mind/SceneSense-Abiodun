"""Sampled replay-minibatch reward_mean vs gradient update (1-10,000), per arm and seed.

``reward_mean`` is the mean reward of the 256-row minibatch sampled for that update (``UpdateMetricsV1``), from
metrics.jsonl (original run) and metrics_resume_01.jsonl (resumed run). Before plotting: every update 1..10,000 must
be present exactly once after merging, and every update present in both files must agree exactly on reward_mean,
critic_loss, actor_loss and target_mean. Raw per-update values are drawn thin and translucent; the labelled smoothed
line is a trailing 200-update rolling mean. The two reward definitions are in separate panels with their own y scale
(never a shared or dual axis).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEED_COLOR = {17: "#2a78d6", 29: "#eb6834", 43: "#1baf7a"}
ARMS = {"CONTROL": "CONTROL arm: current reward (Q_perc - 0.25 L/170; timeout -1)",
        "CANDIDATE": "CANDIDATE arm: Y x (0.7 + 0.3 Q_seg); timeout 0"}
WINDOW = 200
KEYS = ("reward_mean", "critic_loss", "actor_loss", "target_mean")


def merged(run_dir: Path):
    rows, overlap, mismatch = {}, 0, []
    for name in ("metrics.jsonl", "metrics_resume_01.jsonl"):
        seen = set()
        for line in (run_dir / name).read_text().splitlines():
            m = json.loads(line)
            u = m["update"]
            if u in seen:
                raise ValueError(f"{run_dir.name}/{name}: duplicate update {u}")
            seen.add(u)
            if u in rows:
                overlap += 1
                if any(rows[u][k] != m[k] for k in KEYS):
                    mismatch.append(u)
            else:
                rows[u] = m
    complete = sorted(rows) == list(range(1, 10_001))
    return [rows[u]["reward_mean"] for u in range(1, 10_001)] if complete else None, dict(
        complete=complete, overlap_updates=overlap, overlap_mismatches=mismatch)


def main(runs: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    data, verification = {}, {}
    for arm in ARMS:
        for seed in SEED_COLOR:
            values, check = merged(runs / f"{arm}_seed{seed}")
            verification[f"{arm}_seed{seed}"] = check
            if not check["complete"] or check["overlap_mismatches"]:
                raise ValueError(f"{arm} seed {seed}: {check}")
            data[(arm, seed)] = np.asarray(values)
    x = np.arange(1, 10_001)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, facecolor=SURFACE)
    for ax, (arm, title) in zip(axes, ARMS.items()):
        ax.set_facecolor(SURFACE)
        ends = []
        for seed, color in SEED_COLOR.items():
            y = data[(arm, seed)]
            ax.plot(x, y, color=color, lw=0.5, alpha=0.18)
            smooth = np.convolve(y, np.ones(WINDOW) / WINDOW, mode="valid")
            ax.plot(x[WINDOW - 1:], smooth, color=color, lw=2, label=f"seed {seed} ({WINDOW}-update rolling mean)")
            ends.append([float(smooth[-1]), seed])
        lo, hi = ax.get_ylim()
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06 * (hi - lo))
        for yv, seed in ends:
            ax.annotate(f"seed {seed}", (10_000, yv), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8,
                        va="center", annotation_clip=False)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("Replay minibatch reward_mean", color=INK, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlim(0, 10_700)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2,
                  title="thin translucent = raw per-update value", title_fontsize=8)
    axes[-1].set_xlabel("Gradient update", color=INK2, fontsize=9)
    fig.suptitle("Restricted Run-5 pilot: sampled replay-minibatch reward vs update (256-row batches, updates 1-10,000)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "REPLAY_REWARD.png", dpi=130, facecolor=SURFACE)
    with open(out / "REPLAY_REWARD.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "update", "reward_mean"])
        for (arm, seed), y in sorted(data.items()):
            for u, v in zip(x, y):
                w.writerow([arm, seed, int(u), repr(float(v))])
    (out / "REPLAY_REWARD_VERIFICATION.json").write_text(json.dumps(verification, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
