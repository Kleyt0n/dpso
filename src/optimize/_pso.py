from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class PSOState(NamedTuple):
    positions: jnp.ndarray    # (N, D)
    velocities: jnp.ndarray   # (N, D)
    p_best: jnp.ndarray       # (N, D)
    p_best_fit: jnp.ndarray   # (N,)
    g_best: jnp.ndarray       # (D,)
    g_best_fit: jnp.ndarray   # scalar


@dataclass(frozen=True)
class PSOConfig:
    n_particles: int
    n_dims: int
    max_iters: int
    omega: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    lb: float = -5.12
    ub: float = 5.12
    v_max: float | None = None

    @property
    def effective_v_max(self) -> float:
        return self.v_max if self.v_max is not None else 0.2 * (self.ub - self.lb)


def init_swarm(
    config: PSOConfig,
    obj_fn: Callable,
    key: jax.Array,
) -> PSOState:
    k1, k2 = jax.random.split(key)
    positions = jax.random.uniform(
        k1,
        shape=(config.n_particles, config.n_dims),
        minval=config.lb,
        maxval=config.ub,
    )
    velocities = jnp.zeros_like(positions)
    fitness = jax.vmap(obj_fn)(positions)
    best_idx = jnp.argmin(fitness)
    return PSOState(
        positions=positions,
        velocities=velocities,
        p_best=positions,
        p_best_fit=fitness,
        g_best=positions[best_idx],
        g_best_fit=fitness[best_idx],
    )


def pso_step(
    state: PSOState,
    key: jax.Array,
    config: PSOConfig,
    obj_fn: Callable,
) -> PSOState:
    k1, k2 = jax.random.split(key)
    N, D = state.positions.shape
    v_max = config.effective_v_max

    r1 = jax.random.uniform(k1, shape=(N, D))
    r2 = jax.random.uniform(k2, shape=(N, D))

    v_new = (
        config.omega * state.velocities
        + config.c1 * r1 * (state.p_best - state.positions)
        + config.c2 * r2 * (state.g_best - state.positions)
    )
    v_new = jnp.clip(v_new, -v_max, v_max)
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


def pso_run(
    config: PSOConfig,
    obj_fn: Callable,
    key_init: jax.Array,
    key_run: jax.Array,
) -> tuple[PSOState, jnp.ndarray]:
    state = init_swarm(config, obj_fn, key_init)

    keys = jax.random.split(key_run, config.max_iters)

    def scan_fn(state: PSOState, key: jax.Array):
        new_state = pso_step(state, key, config, obj_fn)
        return new_state, new_state.g_best_fit

    final_state, history = jax.lax.scan(scan_fn, state, keys)
    return final_state, history
