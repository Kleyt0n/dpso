from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()

# NeurIPS-ready matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

COLORS = {
    "ours":        "#c4553a",   # DPSO — rust red
    "baseline":    "#3d5a80",   # PSO  — slate blue
    "ours_bg":     "#f5e3df",   # DPSO fill
    "baseline_bg": "#dfe6ed",   # PSO fill
}

PSO_STYLE  = {"color": COLORS["baseline"], "linestyle": "--", "label": "PSO"}
DPSO_STYLE = {"color": COLORS["ours"],     "linestyle": "-",  "label": "DPSO"}
PSO_FILL   = COLORS["baseline_bg"]
DPSO_FILL  = COLORS["ours_bg"]


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_summary_table(results: dict[str, dict]) -> None:
    table = Table(title="Experiment Results", show_lines=True)
    table.add_column("Function", style="bold")
    table.add_column("D", justify="right")
    table.add_column("PSO (mean\u00b1std)", justify="right")
    table.add_column("DPSO (mean\u00b1std)", justify="right")
    table.add_column("Winner", justify="center")

    for key, r in sorted(results.items()):
        pso_mean, pso_std = r["pso_mean"], r["pso_std"]
        dpso_mean, dpso_std = r["dpso_mean"], r["dpso_std"]
        winner = "[green]DPSO[/green]" if dpso_mean < pso_mean else "[red]PSO[/red]"
        table.add_row(
            r["func_name"],
            str(r["dim"]),
            f"{pso_mean:.4e} \u00b1 {pso_std:.4e}",
            f"{dpso_mean:.4e} \u00b1 {dpso_std:.4e}",
            winner,
        )
    console.print(table)


def _fmt_sci_inner(val: float) -> str:
    """Return raw math content (no surrounding $...$)."""
    if val == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    coeff = val / 10**exp
    if exp == 0:
        return f"{coeff:.2f}"
    return f"{coeff:.2f}\\!\\times\\!10^{{{exp}}}"


def _fmt_sci(val: float) -> str:
    return f"${_fmt_sci_inner(val)}$"


def _fmt_pm(mean: float, std: float) -> str:
    return f"${_fmt_sci_inner(mean)} \\pm {_fmt_sci_inner(std)}$"


def _generate_results_subtable(
    results: dict[str, dict],
    dimensions: list[int],
    func_names: list[str],
    caption: str,
    label: str,
) -> str:
    n_dims = len(dimensions)
    col_spec = "l " + "cc " * n_dims
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec.strip()}}}",
        r"\toprule",
    ]

    # Dimension header row
    header_dims = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$D={d}$}}" for d in dimensions
    )
    lines.append(r"\rowcolor{cGroupHd}")
    lines.append(rf"Function & {header_dims} \\")

    # Per-pair cmidrules
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(n_dims)
    )
    lines.append(cmidrules)

    # Sub-header row
    sub_header = " & ".join("PSO & DPSO" for _ in dimensions)
    lines.append(r"\rowcolor{cGroupHd}")
    lines.append(rf" & {sub_header} \\")
    lines.append(r"\midrule")

    # Data rows
    for fname in func_names:
        cells = [fname]
        for d in dimensions:
            key = f"{fname}_{d}"
            r = results[key]
            pso_str = _fmt_pm(r["pso_mean"], r["pso_std"])
            dpso_str = _fmt_pm(r["dpso_mean"], r["dpso_std"])
            if r["dpso_mean"] < r["pso_mean"]:
                dpso_str = rf"\cellcolor{{cBest}}\textbf{{{dpso_str}}}"
            else:
                pso_str = rf"\cellcolor{{cBest}}\textbf{{{pso_str}}}"
            cells.extend([pso_str, dpso_str])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_results_table(
    results: dict[str, dict], dimensions: list[int]
) -> tuple[str, str]:
    from src.utils._functions import BENCHMARKS

    unimodal = [b.name for b in BENCHMARKS if b.group == "unimodal"]
    multimodal = [b.name for b in BENCHMARKS if b.group == "multimodal"]
    # Filter to only functions present in results
    unimodal = [n for n in unimodal if any(f"{n}_{d}" in results for d in dimensions)]
    multimodal = [n for n in multimodal if any(f"{n}_{d}" in results for d in dimensions)]

    uni_tex = _generate_results_subtable(
        results, dimensions, unimodal,
        caption=r"Best fitness values on \textbf{unimodal} benchmarks (mean $\pm$ std over 30 runs). \colorbox{cBest}{\textbf{Bold}} indicates the better result.",
        label="tab:results-unimodal",
    )
    multi_tex = _generate_results_subtable(
        results, dimensions, multimodal,
        caption=r"Best fitness values on \textbf{multimodal} benchmarks (mean $\pm$ std over 30 runs). \colorbox{cBest}{\textbf{Bold}} indicates the better result.",
        label="tab:results-multimodal",
    )
    return uni_tex, multi_tex


def generate_timing_table(results: dict[str, dict], dimensions: list[int]) -> str:
    func_names = []
    seen: set[str] = set()
    for key in sorted(results.keys()):
        name = results[key]["func_name"]
        if name not in seen:
            func_names.append(name)
            seen.add(name)

    n_dims = len(dimensions)
    col_spec = "l " + "cc " * n_dims
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wall-clock time in seconds (mean over 30 runs).}",
        r"\label{tab:timing}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{tabular}}{{{col_spec.strip()}}}",
        r"\toprule",
    ]

    # Dimension header row
    header_dims = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$D={d}$}}" for d in dimensions
    )
    lines.append(r"\rowcolor{cGroupHd}")
    lines.append(rf"Function & {header_dims} \\")

    # Per-pair cmidrules
    cmidrules = "".join(
        rf"\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(n_dims)
    )
    lines.append(cmidrules)

    # Sub-header row
    sub_header = " & ".join("PSO & DPSO" for _ in dimensions)
    lines.append(r"\rowcolor{cGroupHd}")
    lines.append(rf" & {sub_header} \\")
    lines.append(r"\midrule")

    for fname in func_names:
        cells = [fname]
        for d in dimensions:
            key = f"{fname}_{d}"
            r = results[key]
            cells.extend([f"${r['pso_time_mean']:.2f}$", f"${r['dpso_time_mean']:.2f}$"])
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def plot_convergence(
    results: dict[str, dict],
    func_name: str,
    dimensions: list[int],
    output_dir: Path = Path("paper/outputs/figures"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.5, 1.8), squeeze=False)

    for i, d in enumerate(dimensions):
        ax = axes[0, i]
        key = f"{func_name}_{d}"
        r = results[key]

        pso_hist = np.array(r["pso_histories"])   # (n_runs, T)
        dpso_hist = np.array(r["dpso_histories"])  # (n_runs, T)
        iters = np.arange(1, pso_hist.shape[1] + 1)

        for hist, style, fill_c in [
            (pso_hist, PSO_STYLE, PSO_FILL),
            (dpso_hist, DPSO_STYLE, DPSO_FILL),
        ]:
            median = np.median(hist, axis=0)
            q25 = np.percentile(hist, 25, axis=0)
            q75 = np.percentile(hist, 75, axis=0)
            ax.semilogy(iters, median, **style)
            ax.fill_between(iters, q25, q75, alpha=0.40, color=fill_c, linewidth=0)

        ax.set_title(f"$D={d}$", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        if i == 0:
            ax.set_ylabel("Best fitness (log scale)", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle(func_name, fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"{func_name.lower()}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_convergence_grid(
    results: dict[str, dict],
    dimensions: list[int],
    func_names: list[str],
    output_dir: Path = Path("paper/outputs/figures"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_funcs = len(func_names)
    n_dims = len(dimensions)
    fig, axes = plt.subplots(n_funcs, n_dims, figsize=(5.5, 1.6 * n_funcs), squeeze=False)

    for row, fname in enumerate(func_names):
        for col, d in enumerate(dimensions):
            ax = axes[row, col]
            key = f"{fname}_{d}"
            r = results[key]

            pso_hist = np.array(r["pso_histories"])
            dpso_hist = np.array(r["dpso_histories"])
            iters = np.arange(1, pso_hist.shape[1] + 1)

            for hist, style, fill_c in [
                (pso_hist, PSO_STYLE, PSO_FILL),
                (dpso_hist, DPSO_STYLE, DPSO_FILL),
            ]:
                median = np.median(hist, axis=0)
                q25 = np.percentile(hist, 25, axis=0)
                q75 = np.percentile(hist, 75, axis=0)
                ax.semilogy(iters, median, **style)
                ax.fill_between(iters, q25, q75, alpha=0.40, color=fill_c, linewidth=0)

            if row == 0:
                ax.set_title(f"$D={d}$", fontsize=8)
            if col == 0:
                ax.set_ylabel(fname, fontsize=8)
            if row == n_funcs - 1:
                ax.set_xlabel("Iteration", fontsize=7)
            ax.tick_params(labelsize=6)

    axes[0, -1].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_grid.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_contour(ax, bench_fn, lb, ub, *, n_pts=300):
    """Draw filled contour of bench_fn evaluated at D=2 on *ax*."""
    x = np.linspace(lb, ub, n_pts)
    y = np.linspace(lb, ub, n_pts)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[float(bench_fn(jnp.array([xi, yi])))
                    for xi, yi in zip(xr, yr)]
                   for xr, yr in zip(X, Y)])
    # Log-scale to normalise visual across functions with different ranges
    Z = np.log1p(Z - Z.min())
    ax.contourf(X, Y, Z, levels=30, cmap="coolwarm")
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$", fontsize=7)
    ax.set_ylabel("$x_2$", fontsize=7)
    ax.tick_params(labelsize=6)


# Zoomed contour bounds for functions whose full domain hides structure
_CONTOUR_BOUNDS: dict[str, tuple[float, float]] = {
    "Griewank": (-20.0, 20.0),
    "Schwefel1.2": (-10.0, 10.0),
    "Schwefel2.20": (-10.0, 10.0),
    "Schwefel2.21": (-10.0, 10.0),
    "ChungReynolds": (-10.0, 10.0),
    "Cigar": (-10.0, 10.0),
    "Qing": (-10.0, 10.0),
    "RotHyperEllipsoid": (-10.0, 10.0),
    "Bohachevsky": (-10.0, 10.0),
    "Salomon": (-10.0, 10.0),
    "Pathological": (-10.0, 10.0),
    "SchafferF6": (-10.0, 10.0),
    "Whitley": (-5.0, 5.0),
}


def plot_convergence_grid_group(
    results: dict[str, dict],
    dimensions: list[int],
    benchmarks: list,
    output_path: Path,
) -> None:
    """Plot a convergence grid with a leading contour-plot column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_funcs = len(benchmarks)
    n_cols = 1 + len(dimensions)  # contour + one col per dimension
    fig, axes = plt.subplots(
        n_funcs, n_cols,
        figsize=(7.0, 1.6 * n_funcs),
        squeeze=False,
        gridspec_kw={"width_ratios": [1] * n_cols},
    )

    for row, bench in enumerate(benchmarks):
        # Column 0: contour plot
        ax_contour = axes[row, 0]
        lb, ub = _CONTOUR_BOUNDS.get(bench.name, (bench.lb, bench.ub))
        _plot_contour(ax_contour, bench.fn, lb, ub)
        if row == 0:
            ax_contour.set_title("Landscape", fontsize=8)
        ax_contour.set_ylabel(bench.name, fontsize=8)

        # Columns 1+: convergence curves
        for col_off, d in enumerate(dimensions):
            ax = axes[row, 1 + col_off]
            key = f"{bench.name}_{d}"
            r = results[key]

            pso_hist = np.array(r["pso_histories"])
            dpso_hist = np.array(r["dpso_histories"])
            iters = np.arange(1, pso_hist.shape[1] + 1)

            for hist, style, fill_c in [
                (pso_hist, PSO_STYLE, PSO_FILL),
                (dpso_hist, DPSO_STYLE, DPSO_FILL),
            ]:
                median = np.median(hist, axis=0)
                q25 = np.percentile(hist, 25, axis=0)
                q75 = np.percentile(hist, 75, axis=0)
                ax.semilogy(iters, median, **style)
                ax.fill_between(iters, q25, q75, alpha=0.40, color=fill_c, linewidth=0)

            if row == 0:
                ax.set_title(f"$D={d}$", fontsize=8)
            if row == n_funcs - 1:
                ax.set_xlabel("Iteration", fontsize=7)
            ax.tick_params(labelsize=6)

    axes[0, -1].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(
    results: dict[str, dict],
    dimensions: list[int],
    benchmarks: list,
    output_path: Path,
) -> None:
    """Fitness-vs-time scatter: one panel per D, two points per function.

    X-axis: mean wall-clock time (seconds) over 30 runs.
    Y-axis: mean best fitness (log scale).
    Color: PSO (slate blue) vs DPSO (rust red).
    """
    from matplotlib.lines import Line2D

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(2.3 * n_dims + 0.4, 2.8), squeeze=False)

    for col, d in enumerate(dimensions):
        ax = axes[0, col]
        for b in benchmarks:
            r = results[f"{b.name}_{d}"]
            ax.plot(r["pso_time_mean"], r["pso_mean"], "o",
                    color=COLORS["baseline"], markersize=3.5, alpha=0.8)
            ax.plot(r["dpso_time_mean"], r["dpso_mean"], "o",
                    color=COLORS["ours"], markersize=3.5, alpha=0.8)

        ax.set_yscale("log")
        ax.set_title(f"$D = {d}$", fontsize=9)
        if col == 0:
            ax.set_ylabel("Best fitness", fontsize=8)
        ax.set_xlabel("Wall-clock time (s)", fontsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linewidth=0.3, alpha=0.4)
        ax.tick_params(labelsize=7)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["baseline"],
               markersize=5, label="PSO"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["ours"],
               markersize=5, label="DPSO"),
    ]
    axes[0, -1].legend(handles=handles, fontsize=6, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_csv(
    results: dict[str, dict],
    output_dir: Path = Path("paper/outputs"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.csv"
    fields = [
        "func_name", "dim",
        "pso_mean", "pso_std", "pso_time_mean",
        "dpso_mean", "dpso_std", "dpso_time_mean",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(results.keys()):
            writer.writerow(results[key])


def save_raw_results(
    results: dict[str, dict],
    output_dir: Path = Path("paper/outputs"),
) -> Path:
    """Save raw experiment results (scalars as JSON, histories as .npz).

    Returns the timestamped directory path.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    raw_dir = output_dir / "raw" / ts
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Scalar metadata (everything except history arrays)
    scalars: dict[str, dict] = {}
    history_arrays: dict[str, Any] = {}
    for key in sorted(results.keys()):
        r = results[key]
        scalars[key] = {
            k: v for k, v in r.items()
            if k not in ("pso_histories", "dpso_histories")
        }
        history_arrays[f"{key}__pso"] = np.asarray(r["pso_histories"])
        history_arrays[f"{key}__dpso"] = np.asarray(r["dpso_histories"])

    with open(raw_dir / "results.json", "w") as f:
        json.dump(scalars, f, indent=2)

    np.savez_compressed(raw_dir / "histories.npz", **history_arrays)

    return raw_dir


def load_raw_results(raw_dir: Path) -> dict[str, dict]:
    """Load raw results saved by :func:`save_raw_results`."""
    with open(raw_dir / "results.json") as f:
        scalars = json.load(f)

    data = np.load(raw_dir / "histories.npz")

    results: dict[str, dict] = {}
    for key, r in scalars.items():
        r["pso_histories"] = data[f"{key}__pso"]
        r["dpso_histories"] = data[f"{key}__dpso"]
        results[key] = r

    return results
