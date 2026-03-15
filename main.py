from __future__ import annotations

import argparse
import time
from itertools import groupby
from operator import attrgetter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.optimize import DPSOConfig, PSOConfig, dpso_run, pso_run
from src.utils import (
    BENCHMARKS,
    generate_results_table,
    generate_timing_table,
    load_raw_results,
    make_progress,
    plot_convergence,
    plot_convergence_grid,
    plot_convergence_grid_group,
    plot_tradeoff,
    print_summary_table,
    save_csv,
    save_raw_results,
)
from src.utils._artifacts import console

DIMENSIONS = [10, 30, 50]
N_PARTICLES = 40
MAX_ITERS = 1000
N_RUNS = 30
SEED = 42


def run_experiments(output_dir: Path) -> dict[str, dict]:
    master_key = jax.random.key(SEED)
    results: dict[str, dict] = {}

    total_tasks = len(BENCHMARKS) * len(DIMENSIONS)
    progress = make_progress()

    with progress:
        task_id = progress.add_task("Running experiments...", total=total_tasks)

        for bench in BENCHMARKS:
            for dim in DIMENSIONS:
                key_label = f"{bench.name}_{dim}"
                progress.update(task_id, description=f"{bench.name} D={dim}")

                pso_cfg = PSOConfig(
                    n_particles=N_PARTICLES,
                    n_dims=dim,
                    max_iters=MAX_ITERS,
                    lb=bench.lb,
                    ub=bench.ub,
                )
                dpso_cfg = DPSOConfig(
                    n_particles=N_PARTICLES,
                    n_dims=dim,
                    max_iters=MAX_ITERS,
                    lb=bench.lb,
                    ub=bench.ub,
                    c3=1.0,
                    beta=0.1,
                )

                pso_fits, dpso_fits = [], []
                pso_hists, dpso_hists = [], []
                pso_times, dpso_times = [], []

                for run_i in range(N_RUNS):
                    master_key, k_init, k_pso, k_dpso = jax.random.split(master_key, 4)

                    # PSO
                    t0 = time.perf_counter()
                    state_pso, hist_pso = pso_run(pso_cfg, bench.fn, k_init, k_pso)
                    jax.block_until_ready(state_pso.g_best_fit)
                    pso_times.append(time.perf_counter() - t0)
                    pso_fits.append(float(state_pso.g_best_fit))
                    pso_hists.append(np.asarray(hist_pso))

                    # DPSO — same init, different iteration keys
                    t0 = time.perf_counter()
                    state_dpso, hist_dpso = dpso_run(dpso_cfg, bench.fn, k_init, k_dpso)
                    jax.block_until_ready(state_dpso.g_best_fit)
                    dpso_times.append(time.perf_counter() - t0)
                    dpso_fits.append(float(state_dpso.g_best_fit))
                    dpso_hists.append(np.asarray(hist_dpso))

                results[key_label] = {
                    "func_name": bench.name,
                    "dim": dim,
                    "pso_mean": float(np.mean(pso_fits)),
                    "pso_std": float(np.std(pso_fits)),
                    "dpso_mean": float(np.mean(dpso_fits)),
                    "dpso_std": float(np.std(dpso_fits)),
                    "pso_time_mean": float(np.mean(pso_times)),
                    "dpso_time_mean": float(np.mean(dpso_times)),
                    "pso_histories": np.stack(pso_hists),
                    "dpso_histories": np.stack(dpso_hists),
                }
                progress.advance(task_id)

    raw_dir = save_raw_results(results, output_dir)
    console.print(f"[bold]Raw results saved to {raw_dir}")
    return results


def generate_artifacts(results: dict[str, dict], output_dir: Path) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Summary")
    print_summary_table(results)

    # Individual convergence plots
    console.print("\n[bold]Saving convergence plots...")
    func_names = [b.name for b in BENCHMARKS]
    for fname in func_names:
        plot_convergence(results, fname, DIMENSIONS, output_dir / "figures")
    plot_convergence_grid(results, DIMENSIONS, func_names, output_dir / "figures")

    # Grouped convergence grids with contour column — one grid per group chunk
    for group_name, group_iter in groupby(BENCHMARKS, key=attrgetter("group")):
        group_benchmarks = list(group_iter)
        for chunk_idx in range(0, len(group_benchmarks), 6):
            chunk = group_benchmarks[chunk_idx : chunk_idx + 6]
            suffix = f"_{chunk_idx // 6 + 1}" if len(group_benchmarks) > 6 else ""
            plot_convergence_grid_group(
                results,
                DIMENSIONS,
                chunk,
                output_dir / "figures" / f"convergence_{group_name}{suffix}.pdf",
            )
    # Accuracy-vs-time tradeoff (one plot per group)
    for group_name, group_iter in groupby(BENCHMARKS, key=attrgetter("group")):
        plot_tradeoff(
            results,
            DIMENSIONS,
            list(group_iter),
            output_dir / "figures" / f"tradeoff_{group_name}.pdf",
        )
    console.print(f"  Saved to {output_dir / 'figures'}/")

    # LaTeX tables
    console.print("[bold]Saving LaTeX tables...")
    results_uni_tex, results_multi_tex = generate_results_table(results, DIMENSIONS)
    timing_tex = generate_timing_table(results, DIMENSIONS)
    (output_dir / "tables" / "results_unimodal.tex").write_text(results_uni_tex)
    (output_dir / "tables" / "results_multimodal.tex").write_text(results_multi_tex)
    (output_dir / "tables" / "timing.tex").write_text(timing_tex)
    console.print(f"  Saved to {output_dir / 'tables'}/")

    # CSV
    save_csv(results, output_dir)
    console.print(f"  CSV saved to {output_dir / 'results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPSO experiments")
    parser.add_argument(
        "--from-raw",
        type=str,
        metavar="TIMESTAMP",
        help="Skip experiments; regenerate artifacts from paper/outputs/raw/TIMESTAMP",
    )
    args = parser.parse_args()

    output_dir = Path("paper/outputs")

    if args.from_raw:
        raw_dir = output_dir / "raw" / args.from_raw
        if not raw_dir.exists():
            console.print(f"[red]Not found: {raw_dir}")
            raise SystemExit(1)
        console.print(f"[bold]Loading raw results from {raw_dir}")
        results = load_raw_results(raw_dir)
    else:
        results = run_experiments(output_dir)

    generate_artifacts(results, output_dir)


if __name__ == "__main__":
    main()
