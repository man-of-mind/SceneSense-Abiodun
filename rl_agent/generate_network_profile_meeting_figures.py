#!/usr/bin/env python3
"""Generate reproducible Gaussian/Markov target-SNR meeting figures.

The figures are synthetic design artifacts.  They deliberately describe a
target-SNR trace and never claim measured OAI radio behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/network_profile_design_v2.json"
NORMAL = NormalDist()
STATE_SHORT = ("A", "I", "F")


class DesignError(RuntimeError):
    """Raised when the frozen design contract is internally inconsistent."""


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    route = config["route"]
    duration = float(route["duration_s"])
    step = float(route["sample_period_s"])
    count = int(route["sample_count"])
    if step <= 0 or count <= 0 or not math.isclose(duration / step, count, abs_tol=1e-9):
        raise DesignError("duration/sample-period/sample-count contract is inconsistent")
    target = config["target_snr"]
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    means = np.asarray(target["state_means_db"], dtype=float)
    states = list(target["state_order"])
    if len(states) != 3 or len(set(states)) != 3 or means.shape != (3,):
        raise DesignError("exactly three unique ordered states and means are required")
    if not lower < float(means.min()) < float(means.max()) < upper:
        raise DesignError("state means must lie strictly inside the target-SNR bounds")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_trace_ids: set[str] = set()
    for profile in config["profiles"]:
        profile_id = str(profile["profile_id"])
        seed = int(profile["seed"])
        if profile_id in seen_ids or seed in seen_seeds:
            raise DesignError("profile identifiers and seeds must be unique")
        seen_ids.add(profile_id)
        seen_seeds.add(seed)
        trace_id = str(profile.get("trace_id", f"{profile_id}_SEED_{seed}"))
        if trace_id in seen_trace_ids:
            raise DesignError("profile trace identifiers must be unique")
        seen_trace_ids.add(trace_id)
        matrix = np.asarray(profile["transition_matrix"], dtype=float)
        sigmas = np.asarray(target["state_sigma_db"], dtype=float)
        if matrix.shape != (3, 3) or np.any(matrix < 0):
            raise DesignError(f"{profile_id}: transition matrix must be non-negative 3x3")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
            raise DesignError(f"{profile_id}: every transition row must sum to one")
        if sigmas.shape != (3,) or np.any(sigmas <= 0):
            raise DesignError(f"{profile_id}: three positive state sigmas are required")
        if str(profile["initial_distribution"]) != "STATIONARY":
            raise DesignError(f"{profile_id}: initial distribution must be STATIONARY")


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    augmented = np.vstack((matrix.T - np.eye(matrix.shape[0]), np.ones(matrix.shape[0])))
    rhs = np.concatenate((np.zeros(matrix.shape[0]), np.ones(1)))
    solution = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    solution[np.abs(solution) < 1e-14] = 0.0
    if np.any(solution < -1e-10) or not math.isclose(float(solution.sum()), 1.0, abs_tol=1e-9):
        raise DesignError("could not resolve a valid stationary distribution")
    return solution / solution.sum()


def truncated_normal_pdf(
    x: np.ndarray, mean: float, sigma: float, lower: float, upper: float
) -> np.ndarray:
    alpha = (lower - mean) / sigma
    beta = (upper - mean) / sigma
    normalizer = NORMAL.cdf(beta) - NORMAL.cdf(alpha)
    z = (x - mean) / sigma
    density = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi) * normalizer)
    return np.where((x >= lower) & (x <= upper), density, 0.0)


def truncated_normal_ppf(
    probability: float, mean: float, sigma: float, lower: float, upper: float
) -> float:
    alpha_cdf = NORMAL.cdf((lower - mean) / sigma)
    beta_cdf = NORMAL.cdf((upper - mean) / sigma)
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    mapped = alpha_cdf + probability * (beta_cdf - alpha_cdf)
    return mean + sigma * NORMAL.inv_cdf(mapped)


def truncated_normal_moments(
    mean: float, sigma: float, lower: float, upper: float
) -> tuple[float, float]:
    alpha = (lower - mean) / sigma
    beta = (upper - mean) / sigma
    alpha_pdf = NORMAL.pdf(alpha)
    beta_pdf = NORMAL.pdf(beta)
    normalizer = NORMAL.cdf(beta) - NORMAL.cdf(alpha)
    shift = (alpha_pdf - beta_pdf) / normalizer
    truncated_mean = mean + sigma * shift
    truncated_variance = sigma * sigma * (
        1.0
        + (alpha * alpha_pdf - beta * beta_pdf) / normalizer
        - shift * shift
    )
    return truncated_mean, truncated_variance


def mixture_pdf(
    x: np.ndarray,
    weights: Sequence[float],
    means: Sequence[float],
    sigmas: Sequence[float],
    lower: float,
    upper: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    components = [
        float(weight) * truncated_normal_pdf(x, float(mean), float(sigma), lower, upper)
        for weight, mean, sigma in zip(weights, means, sigmas)
    ]
    return np.sum(components, axis=0), components


class DeterministicTargetSnrSequence:
    """Stateful fixed-seed sequence that can continue for an entire episode."""

    def __init__(self, profile: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.profile = profile
        self.config = config
        target = config["target_snr"]
        self.means = np.asarray(target["state_means_db"], dtype=float)
        self.sigmas = np.asarray(target["state_sigma_db"], dtype=float)
        self.matrix = np.asarray(profile["transition_matrix"], dtype=float)
        self.lower = float(target["lower_bound_db"])
        self.upper = float(target["upper_bound_db"])
        self.stationary = stationary_distribution(self.matrix)
        self.rng = np.random.default_rng(int(profile["seed"]))
        self.state = int(self.rng.choice(3, p=self.stationary))
        self.sample_index = 0

    def next_sample(self) -> tuple[int, float]:
        if self.sample_index:
            self.state = int(self.rng.choice(3, p=self.matrix[self.state]))
        probability = float(self.rng.random())
        value = truncated_normal_ppf(
            probability,
            self.means[self.state],
            self.sigmas[self.state],
            self.lower,
            self.upper,
        )
        self.sample_index += 1
        return self.state, value


def generate_trace(
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    sample_count: int | None = None,
) -> dict[str, Any]:
    route = config["route"]
    target = config["target_snr"]
    matrix = np.asarray(profile["transition_matrix"], dtype=float)
    count = int(route["sample_count"] if sample_count is None else sample_count)
    if count <= 0:
        raise DesignError("sample count must be positive")
    period = float(route["sample_period_s"])
    state_index = np.empty(count, dtype=int)
    snr = np.empty(count, dtype=float)
    sequence = DeterministicTargetSnrSequence(profile, config)
    for index in range(count):
        state_index[index], snr[index] = sequence.next_sample()
    edges = np.arange(count + 1, dtype=float) * period
    times = edges[:-1]
    return {
        "profile": profile,
        "matrix": matrix,
        "stationary": sequence.stationary,
        "state_index": state_index,
        "snr": snr,
        "times": times,
        "edges": edges,
    }


def autocorrelation(values: np.ndarray, lag: int) -> float | None:
    if lag <= 0 or lag >= len(values):
        return None
    left, right = values[:-lag], values[lag:]
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def contiguous_segments(states: np.ndarray, period: float) -> Iterable[tuple[float, float, int]]:
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            yield start * period, index * period, int(states[start])
            start = index


def configure_matplotlib(config: Mapping[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": config["figure"]["font_family"],
            "font.size": 11.5,
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.2,
            "figure.dpi": 150,
            "savefig.dpi": int(config["figure"]["png_dpi"]),
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def design_footer(config: Mapping[str, Any]) -> str:
    target = config["target_snr"]
    return (
        f"Target-SNR design · L={float(target['lower_bound_db']):g} dB, "
        f"U={float(target['upper_bound_db']):g} dB · not measured achieved-OAI SNR"
    )


def trace_footer(config: Mapping[str, Any]) -> str:
    route = config["route"]
    return (
        f"Target-SNR reference trace · fixed seed/trace ID · "
        f"{int(route['sample_count'])} × {1000 * float(route['sample_period_s']):g} ms · "
        "runtime continues to episode end"
    )


def figure_footer(figure: plt.Figure, text: str) -> None:
    figure.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8.5, color="#49515C")


def trace_time_ticks(duration: float) -> np.ndarray:
    return np.linspace(0.0, duration, 8)


def snr_ticks(config: Mapping[str, Any]) -> list[float]:
    target = config["target_snr"]
    return sorted(
        {
            float(target["lower_bound_db"]),
            *map(float, target["state_means_db"]),
            float(target["upper_bound_db"]),
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, Any] = {"facecolor": "white", "bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = dpi
        figure.savefig(output_dir / f"{stem}.{suffix}", **kwargs)
    plt.close(figure)


def state_colors(config: Mapping[str, Any]) -> tuple[str, str, str]:
    palette = config["figure"]["palette"]
    return palette["grey"], palette["orange"], palette["teal"]


def plot_gaussian_mean_shift(config: Mapping[str, Any]) -> plt.Figure:
    target = config["target_snr"]
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    means = list(map(float, target["state_means_db"]))
    sigma = float(target["state_sigma_db"][0])
    states = list(target["state_order"])
    colors = state_colors(config)
    x = np.linspace(lower, upper, 1200)
    figure, axis = plt.subplots(figsize=(11.8, 5.1))
    for state, mean, color in zip(states, means, colors):
        density = truncated_normal_pdf(x, mean, sigma, lower, upper)
        axis.plot(x, density, color=color, linewidth=2.8, label=f"{state.title()}: μ={mean:g} dB")
        axis.fill_between(x, density, color=color, alpha=0.10)
        axis.axvline(mean, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
    axis.axvline(lower, color=config["figure"]["palette"]["warning"], linestyle=":", linewidth=1.6)
    axis.axvline(upper, color=config["figure"]["palette"]["warning"], linestyle=":", linewidth=1.6)
    axis.text(lower + 0.12, axis.get_ylim()[1] * 0.94, "L", color=config["figure"]["palette"]["warning"], weight="bold")
    axis.text(upper - 0.12, axis.get_ylim()[1] * 0.94, "U", color=config["figure"]["palette"]["warning"], weight="bold", ha="right")
    axis.set(
        xlabel="Target SNR, γ (dB)",
        ylabel="Probability density",
        xlim=(lower - 0.45, upper + 0.45),
        title="Gaussian mean shift: μ controls typical SNR; σ controls spread",
    )
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.text(
        0.5,
        0.82,
        rf"$\gamma \sim \mathcal{{TN}}_{{[{lower:g},{upper:g}]}}(\mu,\,{sigma:g}^2)$",
        transform=axis.transAxes,
        ha="center",
        fontsize=14,
        color=config["figure"]["palette"]["navy"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": config["figure"]["palette"]["light"]},
    )
    figure.subplots_adjust(bottom=0.16, top=0.86)
    figure_footer(figure, design_footer(config))
    return figure


def plot_markov_model(config: Mapping[str, Any]) -> plt.Figure:
    palette = config["figure"]["palette"]
    states = list(config["target_snr"]["state_order"])
    means = list(map(float, config["target_snr"]["state_means_db"]))
    colors = state_colors(config)
    figure, axis = plt.subplots(figsize=(11.8, 5.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    centers = (0.17, 0.50, 0.83)
    for center, state, mean, color in zip(centers, states, means, colors):
        box = FancyBboxPatch(
            (center - 0.115, 0.51), 0.23, 0.18,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.8, edgecolor=color, facecolor=color + "1F",
        )
        axis.add_patch(box)
        axis.text(center, 0.615, state.title(), ha="center", va="center", fontsize=13, weight="bold", color=palette["navy"])
        axis.text(center, 0.555, f"Gaussian around μ={mean:g} dB", ha="center", va="center", fontsize=10.5, color=palette["grey"])
        loop = FancyArrowPatch(
            (center - 0.07, 0.705), (center + 0.07, 0.705),
            connectionstyle="arc3,rad=-1.1", arrowstyle="-|>", mutation_scale=14,
            linewidth=1.8, color=color,
        )
        axis.add_patch(loop)
        axis.text(center, 0.83, r"stay: $P_{ii}$", ha="center", fontsize=10.5, color=palette["grey"])
    for left, right in zip(centers[:-1], centers[1:]):
        forward = FancyArrowPatch(
            (left + 0.12, 0.62), (right - 0.12, 0.62),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=palette["deep"],
        )
        backward = FancyArrowPatch(
            (right - 0.12, 0.56), (left + 0.12, 0.56),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=palette["orange"],
        )
        axis.add_patch(forward)
        axis.add_patch(backward)
    formula_boxes = (
        (0.04, r"State memory", r"$\Pr(Z_{k+1}=j\mid Z_k=i)=P_{ij}$"),
        (0.36, r"Gaussian emission", r"$\gamma_k\mid Z_k=i\sim\mathcal{TN}_{[L,U]}(\mu_i,\sigma_i^2)$"),
        (0.69, r"Expected dwell", r"$\mathbb{E}[D_i]=\Delta t/(1-P_{ii})$"),
    )
    for left, heading, formula in formula_boxes:
        axis.add_patch(Rectangle((left, 0.20), 0.27, 0.22, facecolor="#F7F8FA", edgecolor=palette["light"], linewidth=1.2))
        axis.text(left + 0.135, 0.36, heading, ha="center", fontsize=11.5, weight="bold", color=palette["navy"])
        axis.text(left + 0.135, 0.275, formula, ha="center", fontsize=12, color=palette["deep"])
    axis.text(0.5, 0.985, "Markov-modulated Gaussian profile: distribution plus temporal persistence", ha="center", va="top", fontsize=15, weight="bold", color=palette["navy"])
    axis.text(0.5, 0.095, "Large diagonal probabilities create multi-frame fades and recoveries instead of independent 100-ms jumps.", ha="center", fontsize=11, color=palette["grey"])
    figure.subplots_adjust(bottom=0.08, top=0.97, left=0.02, right=0.98)
    figure_footer(figure, design_footer(config))
    return figure


def plot_transition_matrices(traces: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> plt.Figure:
    palette = config["figure"]["palette"]
    cmap = LinearSegmentedColormap.from_list("idcc_blue", ["#FFFFFF", "#CDEFFC", palette["deep"]])
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.0))
    image = None
    for axis, trace in zip(axes.flat, traces):
        profile = trace["profile"]
        matrix = trace["matrix"]
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                text_color = "white" if value >= 0.62 else palette["navy"]
                axis.text(column, row, f"{value:.3f}" if value > 0.98 else f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=11.5, weight="bold" if row == column else "normal")
            axis.add_patch(Rectangle((row - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor=palette["teal"], linewidth=2.0))
        stationary = np.asarray(trace["stationary"])
        axis.set_title(
            f"{profile['display_name']}\nstationary A/I/F = "
            + "/".join(f"{100 * value:.0f}%" for value in stationary),
            color=palette["navy"], fontsize=11.5,
        )
        axis.set_xticks(range(3), STATE_SHORT)
        axis.set_yticks(range(3), STATE_SHORT)
        axis.set_xlabel("Next state")
        axis.set_ylabel("Current state")
        axis.grid(False)
    assert image is not None
    figure.suptitle("Four network profiles differ through state occupancy and persistence", y=0.975, fontsize=16, weight="bold", color=palette["navy"])
    figure.text(0.5, 0.925, "A = adverse   ·   I = intermediate   ·   F = favorable", ha="center", fontsize=9.5, color=palette["grey"])
    figure.subplots_adjust(left=0.075, right=0.96, bottom=0.13, top=0.85, hspace=0.58, wspace=0.30)
    figure_footer(figure, design_footer(config))
    return figure


def plot_marginal_overview(traces: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> plt.Figure:
    target = config["target_snr"]
    palette = config["figure"]["palette"]
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    means = np.asarray(target["state_means_db"], dtype=float)
    x = np.linspace(lower, upper, 1400)
    figure, axis = plt.subplots(figsize=(11.8, 5.2))
    for trace in traces:
        profile = trace["profile"]
        total, _ = mixture_pdf(x, trace["stationary"], means, target["state_sigma_db"], lower, upper)
        expected = float(np.trapz(x * total, x))
        axis.plot(
            x, total, color=profile["color"], linestyle=profile["line_style"],
            linewidth=2.8, label=f"{profile['display_name']}  (E[γ]≈{expected:.1f} dB)",
        )
        axis.fill_between(x, total, color=profile["color"], alpha=0.045)
    axis.axvline(lower, color=palette["warning"], linestyle=":", linewidth=1.5)
    axis.axvline(upper, color=palette["warning"], linestyle=":", linewidth=1.5)
    axis.set(
        xlabel="Target SNR, γ (dB)", ylabel="Long-run probability density",
        xlim=(lower - 0.35, upper + 0.35),
    )
    figure.suptitle("Target-SNR design marginals for the four network profiles", y=0.97, fontsize=16, weight="bold", color=palette["navy"])
    figure.text(0.5, 0.895, "Mid-variable and fade/recovery overlap: same marginal distribution, different temporal memory", ha="center", fontsize=10.5, color=palette["grey"])
    axis.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    figure.subplots_adjust(bottom=0.17, top=0.73)
    figure_footer(figure, design_footer(config))
    return figure


def add_state_background(axis: plt.Axes, trace: Mapping[str, Any], config: Mapping[str, Any], alpha: float = 0.055) -> None:
    period = float(config["route"]["sample_period_s"])
    colors = state_colors(config)
    for start, end, state in contiguous_segments(trace["state_index"], period):
        axis.axvspan(start, end, color=colors[state], alpha=alpha, linewidth=0)


def plot_trace_overview(traces: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> plt.Figure:
    target = config["target_snr"]
    route = config["route"]
    palette = config["figure"]["palette"]
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    duration = float(route["duration_s"])
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), sharex=True, sharey=True)
    for panel_index, (axis, trace) in enumerate(zip(axes.flat, traces)):
        profile, snr, edges = trace["profile"], trace["snr"], trace["edges"]
        add_state_background(axis, trace, config)
        axis.step(edges, np.r_[snr, snr[-1]], where="post", color=profile["color"], linewidth=1.55)
        axis.axhline(float(np.mean(snr)), color=palette["grey"], linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_title(f"{profile['display_name']}  ·  realized mean {np.mean(snr):.1f} dB", color=palette["navy"], fontsize=11.5)
        axis.set_xlim(0, duration)
        axis.set_ylim(lower - 0.4, upper + 0.4)
        axis.set_xticks(trace_time_ticks(duration))
        axis.set_yticks(snr_ticks(config))
        axis.tick_params(axis="x", labelbottom=True)
        if panel_index // 2 == 1:
            axis.set_xlabel("Route-loop time (s)")
        if panel_index % 2 == 0:
            axis.set_ylabel("Target SNR (dB)")
        else:
            axis.tick_params(axis="y", labelleft=True)
    figure.suptitle(
        f"Fixed-seed target-SNR reference at 100 ms across {duration:g} s of Route B ({float(route['route_length_m']):.1f} m)",
        y=0.975, fontsize=15.5, weight="bold", color=palette["navy"],
    )
    figure.text(0.5, 0.91, "Background state: adverse (grey) · intermediate (orange) · favorable (teal)", ha="center", fontsize=9.5, color=palette["grey"])
    figure.subplots_adjust(left=0.085, right=0.98, bottom=0.13, top=0.84, hspace=0.42, wspace=0.13)
    figure_footer(figure, trace_footer(config))
    return figure


def plot_profile_distribution(trace: Mapping[str, Any], config: Mapping[str, Any]) -> plt.Figure:
    target, profile = config["target_snr"], trace["profile"]
    palette = config["figure"]["palette"]
    states = list(target["state_order"])
    means = np.asarray(target["state_means_db"], dtype=float)
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    x = np.linspace(lower, upper, 1400)
    total, components = mixture_pdf(x, trace["stationary"], means, target["state_sigma_db"], lower, upper)
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for state, weight, component, color in zip(states, trace["stationary"], components, state_colors(config)):
        axis.plot(x, component, color=color, linestyle="--", linewidth=1.35, label=f"{state.title()} component ({100 * weight:.0f}%)")
        axis.fill_between(x, component, color=color, alpha=0.07)
    axis.plot(x, total, color=profile["color"], linewidth=3.1, label="Profile mixture")
    axis.set(
        xlabel="Target SNR, γ (dB)", ylabel="Long-run probability density",
        xlim=(lower - 0.35, upper + 0.35), title=f"{profile['display_name']}: stationary target-SNR distribution",
    )
    axis.legend(frameon=False, loc="upper right")
    axis.text(0.02, 0.96, profile["short_description"], transform=axis.transAxes, va="top", fontsize=10, color=palette["grey"])
    figure.subplots_adjust(bottom=0.17, top=0.85)
    figure_footer(figure, design_footer(config))
    return figure


def plot_profile_trace(trace: Mapping[str, Any], config: Mapping[str, Any]) -> plt.Figure:
    target, route, profile = config["target_snr"], config["route"], trace["profile"]
    palette = config["figure"]["palette"]
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    duration = float(route["duration_s"])
    snr, edges = trace["snr"], trace["edges"]
    figure, axis = plt.subplots(figsize=(11.8, 4.5))
    add_state_background(axis, trace, config, alpha=0.07)
    axis.step(edges, np.r_[snr, snr[-1]], where="post", color=profile["color"], linewidth=1.75)
    axis.axhline(float(np.mean(snr)), color=palette["grey"], linestyle="--", linewidth=1.1, label=f"Empirical mean {np.mean(snr):.1f} dB")
    axis.set(
        xlabel="Route-loop time (s)", ylabel="Target SNR (dB)",
        xlim=(0, duration), ylim=(lower - 0.4, upper + 0.4),
        title=f"{profile['display_name']}: {len(snr)} target values over the {duration:g}-s reference trace",
    )
    axis.set_xticks(trace_time_ticks(duration))
    axis.set_yticks(snr_ticks(config))
    axis.legend(frameon=False, loc="upper right")
    axis.text(0.01, 0.94, profile["short_description"], transform=axis.transAxes, va="top", fontsize=10, color=palette["grey"])
    figure.subplots_adjust(bottom=0.19, top=0.82)
    figure_footer(figure, trace_footer(config))
    return figure


def plot_profile_card(trace: Mapping[str, Any], config: Mapping[str, Any]) -> plt.Figure:
    target, route, profile = config["target_snr"], config["route"], trace["profile"]
    palette = config["figure"]["palette"]
    states = list(target["state_order"])
    means = np.asarray(target["state_means_db"], dtype=float)
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    duration = float(route["duration_s"])
    x = np.linspace(lower, upper, 1200)
    total, components = mixture_pdf(x, trace["stationary"], means, target["state_sigma_db"], lower, upper)
    figure, (density_axis, trace_axis) = plt.subplots(1, 2, figsize=(12.2, 5.35), gridspec_kw={"width_ratios": [0.82, 1.75]})
    for state, weight, component, color in zip(states, trace["stationary"], components, state_colors(config)):
        density_axis.plot(x, component, color=color, linestyle="--", linewidth=1.15)
        density_axis.fill_between(x, component, color=color, alpha=0.07, label=f"{state.title()} {100 * weight:.0f}%")
    density_axis.plot(x, total, color=profile["color"], linewidth=2.8, label="Mixture")
    density_axis.set(xlabel="Target SNR (dB)", ylabel="Density", xlim=(lower, upper), title="Long-run distribution")
    density_axis.legend(frameon=False, fontsize=8.2, loc="upper right")
    add_state_background(trace_axis, trace, config, alpha=0.065)
    snr, edges = trace["snr"], trace["edges"]
    trace_axis.step(edges, np.r_[snr, snr[-1]], where="post", color=profile["color"], linewidth=1.55)
    trace_axis.axhline(float(np.mean(snr)), color=palette["grey"], linestyle="--", linewidth=1.0)
    trace_axis.set(
        xlabel="Route-loop time (s)", ylabel="Target SNR (dB)",
        xlim=(0, duration), ylim=(lower - 0.4, upper + 0.4), title=f"Fixed-seed {duration:g}-s reference trace",
    )
    trace_axis.set_xticks(trace_time_ticks(duration))
    trace_axis.set_yticks(snr_ticks(config))
    stationary = np.asarray(trace["stationary"])
    dwell = float(route["sample_period_s"]) / (1.0 - np.diag(trace["matrix"]))
    figure.suptitle(profile["display_name"], y=0.965, fontsize=17, weight="bold", color=palette["navy"])
    figure.text(0.5, 0.895, profile["short_description"], ha="center", fontsize=10.5, color=palette["grey"])
    figure.text(
        0.5, 0.055,
        "Stationary A/I/F = " + "/".join(f"{100 * value:.0f}%" for value in stationary)
        + "   ·   expected state dwell = " + "/".join(f"{value:.1f}s" for value in dwell),
        ha="center", fontsize=9.2, color=palette["grey"],
    )
    figure.subplots_adjust(bottom=0.18, top=0.79, wspace=0.30)
    figure_footer(figure, trace_footer(config))
    return figure


def write_trace_table(output_dir: Path, traces: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    path = output_dir / "traces.csv"
    states = list(config["target_snr"]["state_order"])
    period = float(config["route"]["sample_period_s"])
    fields = (
        "profile_id", "trace_id", "seed", "value_semantics", "step_index",
        "interval_start_s", "interval_end_s", "state_index", "state", "target_snr_db",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trace in traces:
            profile = trace["profile"]
            for index, (state, snr) in enumerate(zip(trace["state_index"], trace["snr"])):
                writer.writerow(
                    {
                        "profile_id": profile["profile_id"],
                        "trace_id": profile.get(
                            "trace_id", f"{profile['profile_id']}_SEED_{profile['seed']}"
                        ),
                        "seed": profile["seed"],
                        "value_semantics": "TARGET_SNR_DESIGN_NOT_MEASURED_ACHIEVED_OAI_SNR",
                        "step_index": index,
                        "interval_start_s": f"{index * period:.1f}",
                        "interval_end_s": f"{(index + 1) * period:.1f}",
                        "state_index": int(state),
                        "state": states[int(state)],
                        "target_snr_db": f"{float(snr):.6f}",
                    }
                )


def build_summaries(traces: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    route = config["route"]
    target = config["target_snr"]
    states = list(target["state_order"])
    period = float(route["sample_period_s"])
    lower = float(target["lower_bound_db"])
    upper = float(target["upper_bound_db"])
    component_moments = [
        truncated_normal_moments(float(mean), float(sigma), lower, upper)
        for mean, sigma in zip(target["state_means_db"], target["state_sigma_db"])
    ]
    component_means = np.asarray([value[0] for value in component_moments])
    component_variances = np.asarray([value[1] for value in component_moments])
    summaries = []
    for trace in traces:
        profile, snr, state_index = trace["profile"], trace["snr"], trace["state_index"]
        stationary = np.asarray(trace["stationary"])
        dwell = period / (1.0 - np.diag(trace["matrix"]))
        occupancy = np.bincount(state_index, minlength=3) / len(state_index)
        transitions = int(np.count_nonzero(state_index[1:] != state_index[:-1]))
        expected_transition_rate = 1.0 - float(np.dot(stationary, np.diag(trace["matrix"])))
        theoretical_mean = float(np.dot(stationary, component_means))
        theoretical_variance = float(
            np.dot(
                stationary,
                component_variances + np.square(component_means - theoretical_mean),
            )
        )
        trace_digest = hashlib.sha256()
        trace_digest.update(np.asarray(state_index, dtype="<i4").tobytes())
        trace_digest.update(np.asarray(snr, dtype="<f8").tobytes())
        summaries.append(
            {
                "profile_id": profile["profile_id"],
                "display_name": profile["display_name"],
                "trace_id": profile.get(
                    "trace_id", f"{profile['profile_id']}_SEED_{profile['seed']}"
                ),
                "seed": int(profile["seed"]),
                "value_semantics": "TARGET_SNR_DESIGN_NOT_MEASURED_ACHIEVED_OAI_SNR",
                "sample_count": len(snr),
                "duration_s": float(route["duration_s"]),
                "theoretical_marginal_mean_snr_db": theoretical_mean,
                "theoretical_marginal_variance_snr_db2": theoretical_variance,
                "empirical_mean_snr_db": float(np.mean(snr)),
                "empirical_variance_snr_db2": float(np.var(snr)),
                "empirical_std_snr_db": float(np.std(snr)),
                "empirical_p05_snr_db": float(np.quantile(snr, 0.05)),
                "empirical_median_snr_db": float(np.median(snr)),
                "empirical_p95_snr_db": float(np.quantile(snr, 0.95)),
                "lag1_autocorrelation": autocorrelation(snr, 1),
                "lag10_1s_autocorrelation": autocorrelation(snr, 10),
                "absolute_jump_p95_db": float(np.quantile(np.abs(np.diff(snr)), 0.95)),
                "state_transition_count": transitions,
                "expected_state_transition_count": expected_transition_rate * (len(snr) - 1),
                **{f"stationary_{state.lower()}_fraction": float(value) for state, value in zip(states, stationary)},
                **{f"empirical_{state.lower()}_fraction": float(value) for state, value in zip(states, occupancy)},
                **{f"expected_{state.lower()}_dwell_s": float(value) for state, value in zip(states, dwell)},
                "trace_sha256": trace_digest.hexdigest(),
            }
        )
    return summaries


def write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = list(summaries[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def supervisor_table_markdown(summaries: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Target-SNR profile mean and variance", "",
        "> Design traces only; not measured achieved-OAI SNR. RFsim actuation requires a separate calibrated mapping.", "",
        "| Network profile | Theoretical mean after truncation (dB) | Theoretical variance after truncation (dB^2) | 420-s empirical mean (dB) | 420-s empirical variance (dB^2) |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['display_name']} | "
            f"{float(summary['theoretical_marginal_mean_snr_db']):.3f} | "
            f"{float(summary['theoretical_marginal_variance_snr_db2']):.3f} | "
            f"{float(summary['empirical_mean_snr_db']):.3f} | "
            f"{float(summary['empirical_variance_snr_db2']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def formulation_markdown(config: Mapping[str, Any], traces: Sequence[Mapping[str, Any]]) -> str:
    target, route = config["target_snr"], config["route"]
    lower, upper = float(target["lower_bound_db"]), float(target["upper_bound_db"])
    period = float(route["sample_period_s"])
    count = int(route["sample_count"])
    duration = float(route["duration_s"])
    means = ", ".join(f"{float(value):.2f}" for value in target["state_means_db"])
    sigmas = ", ".join(f"{float(value):.2f}" for value in target["state_sigma_db"])
    density_durations = route["qualified_density_durations_s"]
    lines = [
        "# Gaussian and Markov network-profile formulation", "",
        "> All values here are **target-SNR design values**, not measured achieved-OAI SNR. RFsim actuation requires a separate calibrated target-to-RFsim mapping; this design does not assume that RFsim accepts target PUSCH SNR directly.", "",
        "## Bounded Gaussian emission", "",
        f"At each `{1000 * period:g} ms` interval, the generator produces one target-SNR value. The qualified design band is `L={lower:g} dB`, `U={upper:g} dB`, with `R={upper - lower:g} dB`.", "",
        "```text",
        "gamma_k | Z_k=i ~ TruncatedNormal(mu_i, sigma_i^2; L, U)",
        "```", "",
        f"The preserved relative rule gives state means A/I/F = `{means} dB` and state sigmas A/I/F = `{sigmas} dB`.", "",
        "The normalized density is:", "",
        "```text",
        "f(gamma) = phi((gamma-mu)/sigma)",
        "           / {sigma [Phi((U-mu)/sigma) - Phi((L-mu)/sigma)]},  L <= gamma <= U",
        "```", "",
        "The summary computes each component's exact mean and variance after truncation, then combines them using the stationary state probabilities. `phi` and `Phi` are the standard-normal PDF and CDF.", "",
        "## Markov state memory", "",
        "Let `Z_k` be ADVERSE, INTERMEDIATE, or FAVORABLE:", "",
        "```text",
        "Pr(Z_(k+1)=j | Z_k=i) = P_ij",
        "```", "",
        f"The transition matrices are unchanged. At `Delta t={period:g} s`, expected state dwell is:", "",
        "```text",
        "E[D_i] = Delta t / (1 - P_ii)",
        "```", "",
        "Stationary state occupancy and the profile marginal satisfy:", "",
        "```text",
        "pi = pi P,     sum_i pi_i = 1",
        "f_profile(gamma) = sum_i pi_i f_i(gamma)",
        "```", "",
        "## Route B reference trace and runtime continuation", "",
        f"Route `{route['route_id']}` is `{float(route['route_length_m']):.2f} m`. Qualified durations are low `{float(density_durations['low']):.2f} s`, medium `{float(density_durations['medium']):.2f} s`, and dense `{float(density_durations['dense']):.2f} s`.", "",
        f"The presentation artifact is a `{duration:g} s` reference prefix containing `{count}` target values:", "",
        "```text",
        f"I_k = [{period:g} k, {period:g} (k+1)),   k = 0,...,{count - 1}",
        "```", "",
        f"The plotted intervals cover `[0,{duration:g})` seconds. The `{duration:g} s` boundary extends the final held value and is not an additional sample.", "",
        "At episode start, instantiate the profile generator with its fixed seed and replay from sample zero. Low and medium consume their matching prefix; dense consumes approximately the full reference prefix. If any episode lasts longer than 420 seconds, the same RNG and Markov state continue deterministically beyond sample 4199 until episode end. The reference length is not a runtime cap.", "",
        "The same profile trace ID and seed are reused across all 72 action profiles and every density. A new random trace must not be generated per action-profile episode.", "",
        "## Preserved profiles", "",
        "| Profile | Trace ID | Seed | Stationary A/I/F | Expected dwell A/I/F | Expected switches / 420-s reference | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for trace in traces:
        profile = trace["profile"]
        stationary = "/".join(f"{100 * value:.0f}%" for value in trace["stationary"])
        dwell = "/".join(f"{value:.1f}s" for value in float(route["sample_period_s"]) / (1.0 - np.diag(trace["matrix"])))
        transition_rate = 1.0 - float(np.dot(trace["stationary"], np.diag(trace["matrix"])))
        expected_switches = transition_rate * (int(route["sample_count"]) - 1)
        trace_id = profile.get("trace_id", f"{profile['profile_id']}_SEED_{profile['seed']}")
        lines.append(f"| `{profile['profile_id']}` | `{trace_id}` | {profile['seed']} | {stationary} | {dwell} | {expected_switches:.0f} | {profile['short_description']} |")
    lines += [
        "", "## Interpretation boundary", "",
        "These are fixed-seed **target-SNR design traces**. They are not measured achieved-OAI traces. Turning a target into RFsim controls remains a separate calibrated mapping step.", "",
    ]
    return "\n".join(lines)


def presenter_markdown(config: Mapping[str, Any], traces: Sequence[Mapping[str, Any]]) -> str:
    route, target = config["route"], config["target_snr"]
    return "\n".join(
        [
            "# Presenter guide — network-profile figures", "",
            "> Label every value as target-SNR design data. These are not measured achieved-OAI traces, and target-to-RFsim actuation remains a separate calibrated mapping.", "",
            "## The story in four slides", "",
            "### Slide 1 — Gaussian mean shift", "",
            f"**Say:** The qualified design band is `{float(target['lower_bound_db']):g}` to `{float(target['upper_bound_db']):g} dB`. The unchanged relative rule places the adverse, intermediate, and favorable means at `{float(target['state_means_db'][0]):.2f}`, `{float(target['state_means_db'][1]):.2f}`, and `{float(target['state_means_db'][2]):.2f} dB`, with sigma `{float(target['state_sigma_db'][0]):.2f} dB`.", "",
            "### Slide 2 — Why add Markov memory?", "",
            "**Say:** Independent Gaussian draws forget the previous 100-ms value. A Markov state adds persistence: the channel normally stays in its current condition and occasionally transitions. The diagonal transition probability directly determines expected dwell time.", "",
            "### Slide 3 — Four preserved profile distributions", "",
            "**Say:** The profiles are not four fixed SNR values. Favorable and adverse change long-run state occupancy. Mid-variable and fade/recovery deliberately have the same long-run SNR distribution, but fade/recovery scales transition rates down by five. This isolates rapid variation from sustained fades.", "",
            "### Slide 4 — Route B reference traces", "",
            f"**Say:** Route B is `{float(route['route_length_m']):.2f} m`. The longest qualified density duration is `{float(route['qualified_density_durations_s']['dense']):.2f} s`, so the presentation uses a `{float(route['duration_s']):g} s`, `{int(route['sample_count'])}`-sample reference trace at 100-ms cadence.", "",
            "**Say:** For fair action-profile comparison, each network profile has one trace ID and fixed seed. Every one of the 72 action profiles and every density restarts that sequence at sample zero. Low and medium stop on their shorter Route B completion; dense uses approximately the full reference. If a run exceeds 420 seconds, the stateful generator continues deterministically until episode end.", "",
            "**Do not say:** that RFsim accepts target PUSCH SNR directly. A calibrated target-to-RFsim mapping is a separate actuation layer.", "",
            "## Figure mapping", "",
            "- `01_gaussian_mean_shift`: explains mean and variance.",
            "- `02_markov_model`: explains temporal memory and dwell time.",
            "- `03_markov_transition_matrices`: technical backup for the four matrices.",
            "- `04_profile_marginal_distributions`: compares long-run profile shapes.",
            "- `05_target_snr_trace_overview`: compares all four 420-s reference traces.",
            "- `profiles/*_card`: one distribution plus its route trace for a dedicated slide.", "",
            "Every figure is available as PNG for convenient insertion and PDF/SVG for vector-quality editing.", "",
        ]
    )


def write_manifest(output_dir: Path, config_path: Path, config: Mapping[str, Any]) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema": "scenesense.network_profile_design_manifest.v1",
            "status": "TARGET_SNR_DESIGN_FIGURES_GENERATED",
            "claim_boundary": config["claim_boundary"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "outputs": files,
        },
    )


def run(config_path: Path, output_dir: Path) -> None:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise DesignError(f"create-only output already exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_dir.mkdir(parents=True)
    atomic_json(output_dir / "resolved_config.json", config)
    configure_matplotlib(config)
    traces = [generate_trace(profile, config) for profile in config["profiles"]]
    write_trace_table(output_dir, traces, config)
    summaries = build_summaries(traces, config)
    write_summary_csv(output_dir / "profile_summary.csv", summaries)
    atomic_json(output_dir / "profile_summary.json", summaries)
    atomic_text(output_dir / "SUPERVISOR_PROFILE_TABLE.md", supervisor_table_markdown(summaries))
    atomic_text(output_dir / "MATHEMATICAL_FORMULATION.md", formulation_markdown(config, traces))
    atomic_text(output_dir / "PRESENTER_GUIDE.md", presenter_markdown(config, traces))
    dpi = int(config["figure"]["png_dpi"])
    figures = output_dir / "figures"
    save_figure(plot_gaussian_mean_shift(config), figures, "01_gaussian_mean_shift", dpi)
    save_figure(plot_markov_model(config), figures, "02_markov_model", dpi)
    save_figure(plot_transition_matrices(traces, config), figures, "03_markov_transition_matrices", dpi)
    save_figure(plot_marginal_overview(traces, config), figures, "04_profile_marginal_distributions", dpi)
    save_figure(plot_trace_overview(traces, config), figures, "05_target_snr_trace_overview", dpi)
    profile_dir = figures / "profiles"
    for index, trace in enumerate(traces, start=1):
        stem = f"{index:02d}_{str(trace['profile']['profile_id']).lower()}"
        save_figure(plot_profile_distribution(trace, config), profile_dir, stem + "_distribution", dpi)
        save_figure(plot_profile_trace(trace, config), profile_dir, stem + "_trace", dpi)
        save_figure(plot_profile_card(trace, config), profile_dir, stem + "_card", dpi)
    write_manifest(output_dir, config_path, config)
    print(json.dumps({"status": "TARGET_SNR_DESIGN_FIGURES_GENERATED", "output_dir": str(output_dir), "profiles": len(traces), "samples_per_profile": int(config["route"]["sample_count"])}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(Path(args.config), Path(args.output_dir))
    except (DesignError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
