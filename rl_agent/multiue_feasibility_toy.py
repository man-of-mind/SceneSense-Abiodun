"""
Minimal multi-UE contention feasibility model (standalone; NOT the full surrogate).
Question: does a COORDINATED policy beat decentralized GREEDY+backoff enough to
justify RL in the multi-UE setting? Grounded in the measured 4-rung capacities.

Honest scope: captures shared-capacity contention + measured collapse (over-capacity
= congestion), lagged decentralized backoff (TCP/AIMD-like), correlated freshness
criticality, and coordinated prioritization. It does NOT reproduce the full reward,
exact OAI collapse shape, or per-UE heterogeneity beyond speed. Directional signal only.
"""
import numpy as np

RUNGS = {"clear": 37.0, "mild": 28.0, "mid": 20.0, "strong": 10.0}
ORDER = ["clear", "mild", "mid", "strong"]
TMAT = {  # measured sticky Markov transition matrix
    "clear":  [0.92, 0.08, 0.00, 0.00],
    "mild":   [0.04, 0.90, 0.06, 0.00],
    "mid":    [0.00, 0.05, 0.90, 0.05],
    "strong": [0.00, 0.00, 0.08, 0.92],
}
PER_SEND_MBPS = 7.4     # one 90KB seg-safe frame @ 10 fps
DT = 0.05               # 20 Hz control
EPS, BASE = 2.0, 1.11
DRIFT = (EPS**2 - BASE**2) ** 0.5   # allowed speed*AoI (m) before ε breach


def max_concurrent(cap_mbps):
    return max(1, int(cap_mbps // PER_SEND_MBPS))


def run(N, lag, correlated, policy, seed, ticks=4000, collapse_frac=1.0):
    rng = np.random.default_rng(seed)
    speeds = rng.uniform(2.0, 10.0, N)                       # m/s mix
    aoi_max = np.maximum(1.0, (DRIFT / speeds) / DT)          # ticks each UE can coast before stale
    time_since = (np.zeros(N) if correlated else rng.uniform(0, 1, N) * aoi_max)
    rung = "mild"
    psend = np.ones(N)                                       # greedy AIMD send-probability
    fail_buf = [np.zeros(N) for _ in range(lag + 1)]         # lagged congestion observation
    fresh = 0; total = 0; collapse = 0
    for t in range(ticks):
        if t > 0:
            rung = ORDER[rng.choice(4, p=TMAT[rung])]
        cap = RUNGS[rung] * rng.uniform(0.7, 1.3)
        K = max_concurrent(cap)
        urgency = time_since / aoi_max                        # >=1 => already stale
        due = time_since >= 0.8 * aoi_max                     # wants to refresh before breach
        if policy == "greedy":
            obs_fail = fail_buf[0]                            # what it saw `lag` steps ago
            psend[obs_fail > 0] *= 0.5                        # back off on observed congestion
            psend[obs_fail == 0] = np.minimum(1.0, psend[obs_fail == 0] + 0.1)
            # critical (already stale) UEs override backoff -- can't afford to defer
            attempt = due & ((rng.random(N) < psend) | (urgency >= 1.0))
        else:  # coordinated oracle: admit the neediest due UEs up to capacity K
            attempt = np.zeros(N, bool)
            didx = np.where(due)[0]
            if len(didx):
                attempt[didx[np.argsort(-urgency[didx])][:K]] = True
        senders = np.where(attempt)[0]
        delivered = np.zeros(N, bool)
        col = False
        if len(senders) <= K:
            delivered[senders] = True
        else:
            # measured congestion collapse: over-offering DESTROYS throughput below K
            eff = max(0, int(round(K * collapse_frac)))
            if eff:
                delivered[rng.choice(senders, eff, replace=False)] = True
            col = True
        fail_buf.append((attempt & ~delivered).astype(float)); fail_buf.pop(0)
        time_since += 1
        time_since[delivered] = 0.0
        fresh += int((time_since < aoi_max).sum()); total += N
        collapse += int(col)
    return fresh / total, collapse / ticks


def sweep():
    seeds = range(30)
    print("collapse_frac = throughput RETAINED when over-subscribed "
          "(1.0=graceful, 0.0=total collapse/measured-harsh). lag=2.")
    print(f"{'N':>3} {'corr':>5} {'collapse_f':>10} | {'greedy_fresh':>12} {'coord_fresh':>11} "
          f"{'coord-greedy(RL headroom)':>26} {'greedy_collapse':>15}")
    print("-" * 92)
    for correlated in (False, True):
        for N in (8, 16):
            for cf in (1.0, 0.5, 0.25, 0.0):
                g = np.mean([run(N, 2, correlated, "greedy", s, collapse_frac=cf) for s in seeds], axis=0)
                c = np.mean([run(N, 2, correlated, "coord", s, collapse_frac=cf) for s in seeds], axis=0)
                gap = c[0] - g[0]
                print(f"{N:>3} {str(correlated):>5} {cf:>10.2f} | {g[0]*100:>11.1f}% {c[0]*100:>10.1f}% "
                      f"{gap*100:>+25.1f}pp {g[1]*100:>14.1f}%")


if __name__ == "__main__":
    sweep()
