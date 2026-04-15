# DPSO: Divergence-Guided Particle Swarm Optimization

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2604.12001)

A JAX-based implementation of DPSO, a PSO variant that adds a divergence-based repulsion term to prevent premature convergence on multimodal landscapes.

## Quick start

```bash
uv sync
uv run python main.py                          # run experiments + generate artifacts
uv run python main.py --from-raw 2026-03-15_143022  # regenerate plots/tables from saved run
```


## How DPSO works

DPSO augments the standard PSO velocity update with a modulation term:

$$
v_{mod} = c_3 * r_3 * \kappa(p_i, g) * \hat{d}
$$

where $\kappa$ is a Gaussian similarity kernel between a particle's personal best and the global best, and $\hat{d}$ points away from the global best. Particles whose personal bests have converged near the global best receive a repulsive push, preserving exploration. Setting $c_3 = 0$ recovers standard PSO.

## License

MIT
