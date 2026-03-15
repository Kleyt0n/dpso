from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from src.optimize._pso import PSOConfig, PSOState, init_swarm


@dataclass(frozen=True)
class DPSOConfig(PSOConfig):
    c3: float = 1.0
    sigma: float | None = None
    beta: float = 0.1

    @property
    def effective_sigma(self) -> float:
        if self.sigma is not None:
            return self.sigma
        return self.beta * jnp.sqrt(float(self.n_dims)) * (self.ub - self.lb)


def compute_v_mod(
    positions: jnp.ndarray,
    p_best: jnp.ndarray,
    g_best: jnp.ndarray,
    c3: float,
    sigma: float,
    r3: jnp.ndarray,
) -> jnp.ndarray:
    eps = 1e-9
    # Kernel: convergence signal from personal bests (Eq. 5)
    p_diff = p_best - g_best
    p_dist = jnp.linalg.norm(p_diff, axis=1, keepdims=True)  # (N, 1)
    kappa = jnp.exp(-p_dist**2 / (2.0 * sigma**2))  # (N, 1)
    # Direction: repulsion from current position (Eq. 6)
    x_diff = positions - g_best  # (N, D)
    x_dist = jnp.linalg.norm(x_diff, axis=1, keepdims=True)  # (N, 1)
    d_hat = x_diff / (x_dist + eps)  # (N, D)
    return c3 * r3 * kappa * d_hat  # (N, D)


def dpso_step(
    state: PSOState,
    key: jax.Array,
    config: DPSOConfig,
    obj_fn: Callable,
) -> PSOState:
    k1, k2, k3 = jax.random.split(key, 3)
    N, D = state.positions.shape
    v_max = config.effective_v_max
    sigma = config.effective_sigma

    r1 = jax.random.uniform(k1, shape=(N, D))
    r2 = jax.random.uniform(k2, shape=(N, D))
    r3 = jax.random.uniform(k3, shape=(N, 1))

    v_std = (
        config.omega * state.velocities
        + config.c1 * r1 * (state.p_best - state.positions)
        + config.c2 * r2 * (state.g_best - state.positions)
    )
    v_mod = compute_v_mod(state.positions, state.p_best, state.g_best, config.c3, sigma, r3)
    v_new = jnp.clip(v_std + v_mod, -v_max, v_max)
    x_new = jnp.clip(state.positions + v_new, config.lb, config.ub)

    fitness = jax.vmap(obj_fn)(x_new)

    improved = fitness < state.p_best_fit
    p_best_new = jnp.where(improved[:, None], x_new, state.p_best)
    p_best_fit_new = jnp.minimum(fitness, state.p_best_fit)

    new_best_idx = jnp.argmin(p_best_fit_new)
    new_g_best_fit = p_best_fit_new[new_best_idx]
    g_best = jnp.where(
        new_g_best_fit < state.g_best_fit,
        p_best_new[new_best_idx],
        state.g_best,
    )
    g_best_fit = jnp.minimum(new_g_best_fit, state.g_best_fit)

    return PSOState(
        positions=x_new,
        velocities=v_new,
        p_best=p_best_new,
        p_best_fit=p_best_fit_new,
        g_best=g_best,
        g_best_fit=g_best_fit,
    )


def dpso_run(
    config: DPSOConfig,
    obj_fn: Callable,
    key_init: jax.Array,
    key_run: jax.Array,
) -> tuple[PSOState, jnp.ndarray]:
    state = init_swarm(config, obj_fn, key_init)

    keys = jax.random.split(key_run, config.max_iters)

    def scan_fn(state: PSOState, key: jax.Array):
        new_state = dpso_step(state, key, config, obj_fn)
        return new_state, new_state.g_best_fit

    final_state, history = jax.lax.scan(scan_fn, state, keys)
    return final_state, history
