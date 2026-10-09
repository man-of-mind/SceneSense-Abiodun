"""Learning curves: per run, rolling means over training decisions (window 2,000 decisions = 500 updates).

Rows: the run's own training reward; timely feedback rate; share of decisions in mode 11; discrete entropy (per
update, rolling over 250 updates). Columns: CONTROL, CANDIDATE. Seeds 17/29/43 = categorical slots 1-3 of the
reference palette (validated all-pairs, light surface), direct-labelled at the line end (contrast relief) and
written as a CSV table. Warm-up decisions (first 288, no updates) are excluded from the rolling windows.
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
ARMS = ("CONTROL", "CANDIDATE")
WINDOW = 2000
WARMUP = 288


def rolling(x, w):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    n = np.cumsum(np.insert(~np.isnan(x), 0, 0))
    out = (c[w:] - c[:-w]) / np.maximum(1, n[w:] - n[:-w])
    return out


def load(run_dir: Path):
    full = run_dir / "decisions_full.jsonl"          # complete log written by the resumed run
    dec = [json.loads(l) for l in (full if full.exists() else run_dir / "decisions.jsonl").read_text().splitlines()][WARMUP:]
    met = {}
    for name in ("metrics.jsonl", "metrics_resume_01.jsonl"):   # resumed updates are bit-identical on the overlap
        if (run_dir / name).exists():
            for line in (run_dir / name).read_text().splitlines():
                m = json.loads(line)
                met[m["update"]] = m
    met = [met[u] for u in sorted(met)]
    if [m["update"] for m in met] != list(range(1, len(met) + 1)):
        raise ValueError(f"{run_dir.name}: metrics are not a contiguous update sequence")
    arm = run_dir.name.split("_")[0]
    reward = [np.nan if d["excluded"] else d["control_reward" if arm == "CONTROL" else "candidate_reward"] for d in dec]
    timely = [float(d["receipt_ms"] is not None and d["receipt_ms"] <= 170.0) for d in dec]
    m11 = [float(d["mode_id"] == 11) for d in dec]
    x_dec = (np.arange(WINDOW, len(dec) + 1)) / 4.0          # 4 decisions per update -> update axis
    ent = [m["discrete_entropy"] for m in met]
    return dict(x_dec=x_dec, reward=rolling(reward, WINDOW), timely=rolling(timely, WINDOW), m11=rolling(m11, WINDOW),
                x_upd=np.arange(250, len(ent) + 1), entropy=rolling(ent, 250))


def plot(runs: Path, out: Path):
    data = {}
    for d in sorted(runs.iterdir()):
        if (d / "RUN_RESULT.json").exists():
            arm, seed = d.name.split("_seed")
            data[(arm, int(seed))] = load(d)
    rows = (("reward", "Training reward (own arm)", "x_dec"), ("timely", "Timely feedback rate", "x_dec"),
            ("m11", "Share of decisions in mode 11", "x_dec"), ("entropy", "Discrete entropy (nats)", "x_upd"))
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 11), sharex=True, facecolor=SURFACE)
    for c, arm in enumerate(ARMS):
        for r, (key, label, xk) in enumerate(rows):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ends = []
            for seed, color in SEED_COLOR.items():
                if (arm, seed) not in data:
                    continue
                v = data[(arm, seed)]
                ax.plot(v[xk], v[key], color=color, lw=2, label=f"seed {seed}")
                ends.append([float(v[key][-1]), seed, float(v[xk][-1])])
            lo, hi = ax.get_ylim()
            gap = 0.07 * (hi - lo)                      # dodge end labels so they never overlap
            ends.sort()
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
            for y, seed, x in ends:
                ax.annotate(f"seed {seed}", (x, y), xytext=(4, 0), textcoords="offset points", color=INK2,
                            fontsize=8, va="center", annotation_clip=False)
            if r == 0 and c == 0:
                ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
            ax.grid(color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=8)
            if c == 0:
                ax.set_ylabel(label, color=INK, fontsize=9)
            if r == 0:
                ax.set_title(f"{arm} reward arm", color=INK, fontsize=11)
            if r == len(rows) - 1:
                ax.set_xlabel("SAC update", color=INK2, fontsize=9)
            ax.set_xlim(0, 10_600)
    fig.suptitle("Restricted Run-5 pilot: learning curves (rolling 2,000 decisions; entropy rolling 250 updates)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "LEARNING_CURVES.png", dpi=130, facecolor=SURFACE)
    with open(out / "LEARNING_CURVES.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "update", "training_reward", "timely_rate", "mode11_share"])
        for (arm, seed), v in sorted(data.items()):
            for i in range(0, len(v["x_dec"]), 500):
                w.writerow([arm, seed, v["x_dec"][i], round(float(v["reward"][i]), 5), round(float(v["timely"][i]), 5),
                            round(float(v["m11"][i]), 5)])


if __name__ == "__main__":
    plot(Path(sys.argv[1]), Path(sys.argv[2]))
