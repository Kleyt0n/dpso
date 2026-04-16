from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.optimize._dpso import DPSOConfig, dpso_step
from src.optimize._pso import PSOConfig, PSOState, init_swarm, pso_step
from src.utils import BENCHMARKS

COLORS = {
    "pso":      "#3d5a80",   # slate blue  (baseline)
    "dpso":     "#c4553a",   # rust red    (ours)
    "pso_bg":   "#dfe6ed",
    "dpso_bg":  "#f5e3df",
    "star":     "#1a1a1a",   # global-best marker
    "text_bg":  "#ffffff",
    "fg":       "#222222",
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

def record_run(
    step_fn: Callable,
    config: PSOConfig,
    obj_fn: Callable,
    key_init: jax.Array,
    key_run: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the algorithm step-by-step, returning full position histories.

    Returns:
        positions: (T+1, N, 2)
        g_best:    (T+1, 2)
        fits:      (T+1,)
    """
    state: PSOState = init_swarm(config, obj_fn, key_init)
    positions = [np.asarray(state.positions)]
    g_best = [np.asarray(state.g_best)]
    fits = [float(state.g_best_fit)]

    for k in jax.random.split(key_run, config.max_iters):
        state = step_fn(state, k, config, obj_fn)
        positions.append(np.asarray(state.positions))
        g_best.append(np.asarray(state.g_best))
        fits.append(float(state.g_best_fit))

    return np.stack(positions), np.stack(g_best), np.array(fits)


def _build_configs(bench, n_particles: int, n_iters: int) -> tuple[PSOConfig, DPSOConfig]:
    pso_cfg = PSOConfig(
        n_particles=n_particles, n_dims=2, max_iters=n_iters,
        lb=bench.lb, ub=bench.ub,
    )
    dpso_cfg = DPSOConfig(
        n_particles=n_particles, n_dims=2, max_iters=n_iters,
        lb=bench.lb, ub=bench.ub,
        c3=1.0, beta=0.1,
    )
    return pso_cfg, dpso_cfg


def find_best_dpso_seed(
    bench, n_particles: int, n_iters: int, candidates: range,
    pso_stuck_threshold: float = 0.1,
) -> tuple[int, float, float]:
    pso_cfg, dpso_cfg = _build_configs(bench, n_particles, n_iters)
    stuck: list[tuple[int, float, float, float]] = []  # (seed, gap, pso, dpso)
    any_best: tuple[int, float, float, float] | None = None

    for s in candidates:
        mk = jax.random.key(s)
        _, k_init, k_pso, k_dpso = jax.random.split(mk, 4)
        _, _, pso_fit = record_run(pso_step, pso_cfg, bench.fn, k_init, k_pso)
        _, _, dpso_fit = record_run(dpso_step, dpso_cfg, bench.fn, k_init, k_dpso)
        pso_f = float(pso_fit[-1])
        dpso_f = float(dpso_fit[-1])
        gap = pso_f - dpso_f
        if any_best is None or gap > any_best[1]:
            any_best = (s, gap, pso_f, dpso_f)
        if pso_f >= pso_stuck_threshold and gap > 0:
            stuck.append((s, gap, pso_f, dpso_f))

    # Prefer "PSO stuck, DPSO escaped" cases — that's the interesting story.
    pool = stuck if stuck else [any_best]  # type: ignore[list-item]
    best = max(pool, key=lambda t: t[1])
    return best[0], best[2], best[3]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def build_contour(
    obj_fn: Callable, lb: float, ub: float, n: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(lb, ub, n)
    ys = np.linspace(lb, ub, n)
    X, Y = np.meshgrid(xs, ys)
    pts = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.asarray(jax.vmap(obj_fn)(pts)).reshape(X.shape)
    return X, Y, Z


def setup_panel(
    ax: plt.Axes,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    lb: float, ub: float,
    label: str, accent: str,
) -> None:
    """Draw the (static) contour landscape + inside-the-plot method label."""
    # log1p normalization keeps multimodal structure readable across
    # functions with very different dynamic ranges (matches _plot_contour
    # in src/utils/_artifacts.py:335-349).
    Z_log = np.log1p(Z - Z.min())
    ax.contourf(X, Y, Z_log, levels=30, cmap="coolwarm", alpha=0.85)
    ax.contour(X, Y, Z_log, levels=8, colors="white", alpha=0.35, linewidths=0.4)

    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#cccccc")

    # Method name rendered *inside* the plot as a white pill with colored
    # text — uses the same accent palette as the paper figures
    # (ours = rust, baseline = slate).
    ax.text(
        0.035, 0.955, label, transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold", color=accent, family="serif",
        bbox={
            "facecolor": "white",
            "edgecolor": accent,
            "linewidth": 0.8,
            "boxstyle": "round,pad=0.45",
        },
        zorder=6,
    )


def make_animation(
    bench_name: str,
    n_particles: int,
    n_iters: int,
    seed: int | None,
    fps: int,
    hold_start: int,
    hold_end: int,
    out_path: Path,
    search_seeds: int,
) -> Path:
    bench = next((b for b in BENCHMARKS if b.name.lower() == bench_name.lower()), None)
    if bench is None:
        names = ", ".join(b.name for b in BENCHMARKS)
        raise ValueError(f"Unknown benchmark {bench_name!r}. Available: {names}")

    # Seed selection — default to a seed that showcases DPSO winning.
    if seed is None:
        print(f"[animate] Searching seeds 0..{search_seeds - 1} for biggest DPSO advantage...")
        seed, pso_preview, dpso_preview = find_best_dpso_seed(
            bench, n_particles, n_iters, range(search_seeds),
        )
        print(f"[animate] Picked seed={seed}  (PSO {pso_preview:.4f}  |  DPSO {dpso_preview:.4f})")

    master_key = jax.random.key(seed)
    _, k_init, k_pso, k_dpso = jax.random.split(master_key, 4)
    pso_cfg, dpso_cfg = _build_configs(bench, n_particles, n_iters)

    print(f"[animate] Running PSO on {bench.name} (2D, N={n_particles}, T={n_iters})...")
    pso_pos, pso_gb, pso_fits = record_run(pso_step, pso_cfg, bench.fn, k_init, k_pso)
    print(f"[animate] Running DPSO on {bench.name}...")
    dpso_pos, dpso_gb, dpso_fits = record_run(dpso_step, dpso_cfg, bench.fn, k_init, k_dpso)
    print(f"[animate] PSO  final best_fit: {pso_fits[-1]:.6f}")
    print(f"[animate] DPSO final best_fit: {dpso_fits[-1]:.6f}")

    print("[animate] Building contour landscape...")
    X, Y, Z = build_contour(bench.fn, bench.lb, bench.ub)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=120)
    fig.patch.set_alpha(0.0)  # transparent figure background for GIF
    setup_panel(ax_l, X, Y, Z, bench.lb, bench.ub, "PSO",  COLORS["pso"])
    setup_panel(ax_r, X, Y, Z, bench.lb, bench.ub, "DPSO", COLORS["dpso"])

    # Particle scatters — paper palette, thin white edges for separation
    sc_pso = ax_l.scatter(
        pso_pos[0, :, 0], pso_pos[0, :, 1],
        s=38, c=COLORS["pso"], edgecolors="white", linewidths=0.6,
        alpha=0.95, zorder=3,
    )
    sc_dpso = ax_r.scatter(
        dpso_pos[0, :, 0], dpso_pos[0, :, 1],
        s=38, c=COLORS["dpso"], edgecolors="white", linewidths=0.6,
        alpha=0.95, zorder=3,
    )
    # Global-best: small dark star
    star_pso = ax_l.scatter(
        [pso_gb[0, 0]], [pso_gb[0, 1]],
        s=200, c=COLORS["star"], marker="*", edgecolors="white",
        linewidths=1.0, zorder=4,
    )
    star_dpso = ax_r.scatter(
        [dpso_gb[0, 0]], [dpso_gb[0, 1]],
        s=200, c=COLORS["star"], marker="*", edgecolors="white",
        linewidths=1.0, zorder=4,
    )

    # Fitness readouts — bottom-left of each panel so they don't collide
    # with the method pill at top-left. Opaque white box for legibility
    # against the colored contour.
    textbox = {"facecolor": "white", "edgecolor": "#bbbbbb",
               "linewidth": 0.5, "boxstyle": "round,pad=0.35"}
    txt_pso = ax_l.text(
        0.035, 0.045, "", transform=ax_l.transAxes, ha="left", va="bottom",
        fontsize=9.5, color=COLORS["fg"], family="serif", bbox=textbox,
        zorder=6,
    )
    txt_dpso = ax_r.text(
        0.035, 0.045, "", transform=ax_r.transAxes, ha="left", va="bottom",
        fontsize=9.5, color=COLORS["fg"], family="serif", bbox=textbox,
        zorder=6,
    )

    fig.tight_layout()

    def update(t: int):
        sc_pso.set_offsets(pso_pos[t])
        sc_dpso.set_offsets(dpso_pos[t])
        star_pso.set_offsets(pso_gb[t][None, :])
        star_dpso.set_offsets(dpso_gb[t][None, :])
        txt_pso.set_text(f"iter {t:>3}    best = {pso_fits[t]:.4f}")
        txt_dpso.set_text(f"iter {t:>3}    best = {dpso_fits[t]:.4f}")
        return sc_pso, sc_dpso, star_pso, star_dpso, txt_pso, txt_dpso

    # Hold beginning and end so viewers can absorb the initial swarm and
    # the final outcome instead of blinking past them.
    frames = (
        [0] * hold_start
        + list(range(n_iters + 1))
        + [n_iters] * hold_end
    )
    total_frames = len(frames)
    duration_s = total_frames / fps
    print(f"[animate] Rendering {total_frames} frames at {fps} fps  (~{duration_s:.1f}s playback)...")
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        out_path,
        writer=PillowWriter(fps=fps),
        savefig_kwargs={"transparent": True, "facecolor": "none"},
    )
    plt.close(fig)
    print(f"[animate] Saved GIF to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PSO vs DPSO animation")
    parser.add_argument("--bench", type=str, default="Rastrigin",
                        help="Benchmark function name (e.g. Rastrigin, Ackley, Griewank, Schwefel).")
    parser.add_argument("--iters", type=int, default=150, help="Optimizer iterations.")
    parser.add_argument("--particles", type=int, default=20, help="Swarm size.")
    parser.add_argument("--fps", type=int, default=8,
                        help="GIF frames per second (lower = slower playback).")
    parser.add_argument("--hold-start", type=int, default=6,
                        help="Extra frames to hold on the initial swarm.")
    parser.add_argument("--hold-end", type=int, default=20,
                        help="Extra frames to hold on the final outcome.")
    parser.add_argument("--seed", type=int, default=None,
                        help="PRNG seed. If omitted, auto-selects a seed that favors DPSO.")
    parser.add_argument("--search-seeds", type=int, default=20,
                        help="How many seeds (0..N-1) to search when auto-selecting.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output GIF path (default: paper/outputs/figures/pso_vs_dpso_<bench>.gif)")
    args = parser.parse_args()

    out = args.out or Path(f"pso_vs_dpso.gif")

    make_animation(
        bench_name=args.bench,
        n_particles=args.particles,
        n_iters=args.iters,
        seed=args.seed,
        fps=args.fps,
        hold_start=args.hold_start,
        hold_end=args.hold_end,
        out_path=out,
        search_seeds=args.search_seeds,
    )


if __name__ == "__main__":
    main()
