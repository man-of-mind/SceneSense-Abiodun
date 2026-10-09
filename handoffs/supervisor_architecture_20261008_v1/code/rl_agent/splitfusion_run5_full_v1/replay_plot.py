"""Training figures for the full Run-5 runs (PNG + CSV; raw values never smoothed silently).

1. REPLAY_REWARD.png: the sampled 256-row replay-minibatch ``reward_mean`` (``UpdateMetricsV1``) against gradient
   update 1-10,000 for seeds 17/29/43, from metrics.jsonl. Before plotting every update 1..10,000 must be present
   exactly once. Raw per-update values thin and translucent; the bold line is a labelled trailing 200-update mean.
2. TRAINING_DECISIONS.png: the decisions collected during training (after the 288-decision warm-up), per
   500-update window: share of each of the 12 modes (one panel per seed) and the timely share. Descriptive of the
   training environment only.
"""
import collections
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
WINDOW = 200
UPDATES = 10_000
WARMUP = 288
FAMILY = ("noAE", "AE128", "AE64", "AE32")
QUANT = ("UINT8", "UINT6", "UINT4")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def load_metrics(run_dir: Path):
    rows = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        m = json.loads(line)
        if m["update"] in rows:
            raise ValueError(f"{run_dir.name}: duplicate update {m['update']}")
        rows[m["update"]] = m
    if sorted(rows) != list(range(1, UPDATES + 1)):
        raise ValueError(f"{run_dir.name}: updates 1..{UPDATES} incomplete")
    return np.array([rows[u]["reward_mean"] for u in range(1, UPDATES + 1)])


def replay_reward(runs: Path, out: Path):
    data = {s: load_metrics(runs / f"seed{s}") for s in SEED_COLOR}
    x = np.arange(1, UPDATES + 1)
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURFACE)
    _style(ax)
    ends = []
    for seed, color in SEED_COLOR.items():
        y = data[seed]
        ax.plot(x, y, color=color, lw=0.5, alpha=0.18)
        smooth = np.convolve(y, np.ones(WINDOW) / WINDOW, mode="valid")
        ax.plot(x[WINDOW - 1:], smooth, color=color, lw=2, label=f"seed {seed} ({WINDOW}-update trailing mean)")
        ends.append([float(smooth[-1]), seed])
    lo, hi = ax.get_ylim()
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05 * (hi - lo))
    for yv, seed in ends:
        ax.annotate(f"seed {seed}", (UPDATES, yv), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8,
                    va="center", annotation_clip=False)
    ax.set_xlim(0, UPDATES * 1.07)
    ax.set_xlabel("Gradient update", color=INK2, fontsize=9)
    ax.set_ylabel("Replay minibatch mean reward", color=INK, fontsize=9)
    ax.set_title("Full Run-5 (12 modes, original q support): sampled 256-row replay-minibatch reward vs update\n"
                 "reward = Y x (0.7 + 0.3 Q_seg) if timely and defined, 0 if late; thin translucent = raw per-update value",
                 color=INK, fontsize=10, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out / "REPLAY_REWARD.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    with open(out / "REPLAY_REWARD.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "update", "reward_mean"])
        for seed, y in sorted(data.items()):
            for u, v in zip(x, y):
                w.writerow([seed, int(u), repr(float(v))])
    return {s: dict(first_200_mean=float(y[:200].mean()), last_200_mean=float(y[-200:].mean())) for s, y in data.items()}


def training_decisions(runs: Path, out: Path, window: int = 500):
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), facecolor=SURFACE, sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 0.8]))
    table = []
    for ax, seed in zip(axes[:3], SEED_COLOR):
        _style(ax)
        rows = [json.loads(l) for l in (runs / f"seed{seed}/decisions.jsonl").read_text().splitlines()]
        actor = [r for r in rows if r["source"] == "STOCHASTIC_ACTOR"]
        per = window * 4                                      # 4 decisions per update
        shares, timely, centers = [], [], []
        for start in range(0, len(actor), per):
            chunk = actor[start:start + per]
            if len(chunk) < per // 2:
                break
            c = collections.Counter(r["mode_id"] for r in chunk)
            shares.append([c.get(m, 0) / len(chunk) for m in range(12)])
            timely.append(sum(r["terminal"] == "SUCCESS" for r in chunk) / len(chunk))
            centers.append((start + len(chunk) / 2) / 4)
            table.append([seed, int(start / 4), int((start + len(chunk)) / 4), timely[-1]] + shares[-1])
        shares = np.array(shares)
        ax.stackplot(centers, shares.T, colors=[colors[m] for m in range(12)], edgecolor=SURFACE, linewidth=0.6,
                     labels=[f"m{m} {FAMILY[m // 3]}/{QUANT[m % 3]}" for m in range(12)])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"seed {seed}\nmode share", color=INK, fontsize=9)
        axes[3].plot(centers, timely, color=SEED_COLOR[seed], lw=2, marker="o", ms=4, label=f"seed {seed}")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8, labelcolor=INK2)
    _style(axes[3])
    axes[3].set_ylabel("timely share\n(receipt <= 170 ms)", color=INK, fontsize=9)
    axes[3].set_ylim(0, 1)
    axes[3].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    axes[3].set_xlabel(f"Gradient update ({window}-update windows of collected stochastic-actor decisions)",
                       color=INK2, fontsize=9)
    axes[0].set_title("Full Run-5 training environment: modes chosen by the stochastic actor and timely share "
                      "(training draws, modeled)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "TRAINING_DECISIONS.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    with open(out / "TRAINING_DECISIONS.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "update_from", "update_to", "timely_share"] + [f"mode{m}_share" for m in range(12)])
        w.writerows(table)


def main(runs: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    summary = replay_reward(runs, out)
    training_decisions(runs, out)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
