from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class BenchmarkFunction:
    name: str
    fn: Callable
    lb: float
    ub: float
    f_opt: float
    group: str = ""


# ---------------------------------------------------------------------------
# Existing benchmark functions
# ---------------------------------------------------------------------------

def _sphere(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)


def _rosenbrock(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _rastrigin(x: jnp.ndarray) -> jnp.ndarray:
    D = x.shape[0]
    return 10.0 * D + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))


def _ackley(x: jnp.ndarray) -> jnp.ndarray:
    D = x.shape[0]
    sum_sq = jnp.sum(x**2) / D
    sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x)) / D
    return -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum_sq)) - jnp.exp(sum_cos) + 20.0 + jnp.e


def _griewank(x: jnp.ndarray) -> jnp.ndarray:
    indices = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    return 1.0 + jnp.sum(x**2) / 4000.0 - jnp.prod(jnp.cos(x / jnp.sqrt(indices)))


def _schwefel(x: jnp.ndarray) -> jnp.ndarray:
    D = x.shape[0]
    return 418.9829 * D - jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))


# ---------------------------------------------------------------------------
# New unimodal functions
# ---------------------------------------------------------------------------

def _sum_squares(x: jnp.ndarray) -> jnp.ndarray:
    """Σ i·xᵢ²"""
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    return jnp.sum(i * x**2)


def _schwefel_2_22(x: jnp.ndarray) -> jnp.ndarray:
    """Σ|xᵢ| + Π|xᵢ|"""
    absx = jnp.abs(x)
    return jnp.sum(absx) + jnp.prod(absx)


def _schwefel_1_2(x: jnp.ndarray) -> jnp.ndarray:
    """Σᵢ(Σⱼ₌₁ⁱ xⱼ)²"""
    return jnp.sum(jnp.cumsum(x) ** 2)


def _schwefel_2_21(x: jnp.ndarray) -> jnp.ndarray:
    """max|xᵢ|"""
    return jnp.max(jnp.abs(x))


def _schwefel_2_20(x: jnp.ndarray) -> jnp.ndarray:
    """Σ|xᵢ|"""
    return jnp.sum(jnp.abs(x))


def _schwefel_2_23(x: jnp.ndarray) -> jnp.ndarray:
    """Σ xᵢ¹⁰"""
    return jnp.sum(x**10)


def _dixon_price(x: jnp.ndarray) -> jnp.ndarray:
    """(x₁-1)² + Σᵢ₌₂ⁿ i(2xᵢ²-xᵢ₋₁)²"""
    i = jnp.arange(2, x.shape[0] + 1, dtype=x.dtype)
    return (x[0] - 1.0) ** 2 + jnp.sum(i * (2.0 * x[1:] ** 2 - x[:-1]) ** 2)


def _zakharov(x: jnp.ndarray) -> jnp.ndarray:
    """Σxᵢ² + (Σ 0.5·i·xᵢ)² + (Σ 0.5·i·xᵢ)⁴"""
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    s = jnp.sum(0.5 * i * x)
    return jnp.sum(x**2) + s**2 + s**4


def _rot_hyper_ellipsoid(x: jnp.ndarray) -> jnp.ndarray:
    """Σ(D-j+1)·xⱼ²"""
    D = x.shape[0]
    w = jnp.arange(D, 0, -1, dtype=x.dtype)
    return jnp.sum(w * x**2)


def _sum_diff_powers(x: jnp.ndarray) -> jnp.ndarray:
    """Σ|xᵢ|^(i+1)"""
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    return jnp.sum(jnp.abs(x) ** (i + 1))


def _chung_reynolds(x: jnp.ndarray) -> jnp.ndarray:
    """(Σxᵢ²)²"""
    return jnp.sum(x**2) ** 2


def _quartic(x: jnp.ndarray) -> jnp.ndarray:
    """Σ i·xᵢ⁴  (De Jong F4 without noise)"""
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    return jnp.sum(i * x**4)


def _cigar(x: jnp.ndarray) -> jnp.ndarray:
    """x₁² + 10⁶·Σᵢ₌₂ⁿ xᵢ²"""
    return x[0] ** 2 + 1e6 * jnp.sum(x[1:] ** 2)


# ---------------------------------------------------------------------------
# New multimodal functions
# ---------------------------------------------------------------------------

def _levy(x: jnp.ndarray) -> jnp.ndarray:
    """Levy function: sin²(πw₁) + Σ(wᵢ-1)²[1+10sin²(πwᵢ+1)] + (wₙ-1)²[1+sin²(2πwₙ)]"""
    w = 1.0 + (x - 1.0) / 4.0
    term1 = jnp.sin(jnp.pi * w[0]) ** 2
    term2 = jnp.sum(
        (w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w[:-1] + 1.0) ** 2)
    )
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + jnp.sin(2.0 * jnp.pi * w[-1]) ** 2)
    return term1 + term2 + term3


def _bohachevsky(x: jnp.ndarray) -> jnp.ndarray:
    """Σᵢ₌₁ⁿ⁻¹[xᵢ²+2xᵢ₊₁²-0.3cos(3πxᵢ)-0.4cos(4πxᵢ₊₁)+0.7]"""
    return jnp.sum(
        x[:-1] ** 2
        + 2.0 * x[1:] ** 2
        - 0.3 * jnp.cos(3.0 * jnp.pi * x[:-1])
        - 0.4 * jnp.cos(4.0 * jnp.pi * x[1:])
        + 0.7
    )


def _salomon(x: jnp.ndarray) -> jnp.ndarray:
    """1 - cos(2π√(Σxᵢ²)) + 0.1√(Σxᵢ²)"""
    r = jnp.sqrt(jnp.sum(x**2))
    return 1.0 - jnp.cos(2.0 * jnp.pi * r) + 0.1 * r


def _alpine1(x: jnp.ndarray) -> jnp.ndarray:
    """Σ|xᵢ sin(xᵢ) + 0.1xᵢ|"""
    return jnp.sum(jnp.abs(x * jnp.sin(x) + 0.1 * x))


def _xin_she_yang2(x: jnp.ndarray) -> jnp.ndarray:
    """(Σ|xᵢ|)·exp(-Σ sin(xᵢ²))"""
    return jnp.sum(jnp.abs(x)) * jnp.exp(-jnp.sum(jnp.sin(x**2)))


def _qing(x: jnp.ndarray) -> jnp.ndarray:
    """Σ(xᵢ²-i)²"""
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    return jnp.sum((x**2 - i) ** 2)


def _pathological(x: jnp.ndarray) -> jnp.ndarray:
    """Σᵢ₌₁ⁿ⁻¹[0.5 + (sin²(√(100xᵢ²+xᵢ₊₁²)) - 0.5) / (1 + 0.001(xᵢ-xᵢ₊₁)⁴)]"""
    num = jnp.sin(jnp.sqrt(100.0 * x[:-1] ** 2 + x[1:] ** 2)) ** 2 - 0.5
    den = 1.0 + 0.001 * (x[:-1] - x[1:]) ** 4
    return jnp.sum(0.5 + num / den)


def _schaffer_f6(x: jnp.ndarray) -> jnp.ndarray:
    """Σᵢ₌₁ⁿ⁻¹[0.5 + (sin²(√(xᵢ²+xᵢ₊₁²)) - 0.5) / (1 + 0.001(xᵢ²+xᵢ₊₁²))²]"""
    ss = x[:-1] ** 2 + x[1:] ** 2
    num = jnp.sin(jnp.sqrt(ss)) ** 2 - 0.5
    den = (1.0 + 0.001 * ss) ** 2
    return jnp.sum(0.5 + num / den)


def _wavy(x: jnp.ndarray) -> jnp.ndarray:
    """1 - (1/n)Σ cos(10xᵢ)exp(-xᵢ²/2)"""
    D = x.shape[0]
    return 1.0 - jnp.sum(jnp.cos(10.0 * x) * jnp.exp(-x**2 / 2.0)) / D


def _weierstrass(x: jnp.ndarray) -> jnp.ndarray:
    """Σᵢ[Σₖ₌₀²⁰ 0.5ᵏ cos(2π·3ᵏ(xᵢ+0.5))] - n·Σₖ₌₀²⁰ 0.5ᵏ cos(π·3ᵏ)"""
    D = x.shape[0]
    k = jnp.arange(0, 21, dtype=x.dtype)
    a = 0.5 ** k
    b = 3.0 ** k
    # Vectorized: sum over k for each xi, then sum over i
    # x shape: (D,), k shape: (21,)
    # broadcast: (D, 1) + (1, 21) -> (D, 21)
    inner = jnp.sum(a[None, :] * jnp.cos(2.0 * jnp.pi * b[None, :] * (x[:, None] + 0.5)), axis=1)
    correction = D * jnp.sum(a * jnp.cos(jnp.pi * b))
    return jnp.sum(inner) - correction


def _pinter(x: jnp.ndarray) -> jnp.ndarray:
    """Σ i·xᵢ² + 20Σ i·sin²(Aᵢ) + Σ i·log₁₀(1+i·Bᵢ²) with wrap-around neighbors"""
    D = x.shape[0]
    i = jnp.arange(1, D + 1, dtype=x.dtype)
    x_prev = jnp.roll(x, 1)  # x_{i-1} with wrap
    x_next = jnp.roll(x, -1)  # x_{i+1} with wrap
    A = x_prev * jnp.sin(x) + jnp.sin(x_next)
    B = x_prev ** 2 - 2.0 * x + 3.0 * x_next - jnp.cos(x) + 1.0
    return (
        jnp.sum(i * x**2)
        + 20.0 * jnp.sum(i * jnp.sin(A) ** 2)
        + jnp.sum(i * jnp.log10(1.0 + i * B**2))
    )


def _stretched_v(x: jnp.ndarray) -> jnp.ndarray:
    """Σ(xᵢ²+xᵢ₊₁²)^0.25·[sin²(50(xᵢ²+xᵢ₊₁²)^0.1)+0.1]"""
    ss = x[:-1] ** 2 + x[1:] ** 2
    return jnp.sum(ss**0.25 * (jnp.sin(50.0 * ss**0.1) ** 2 + 0.1))


def _happy_cat(x: jnp.ndarray) -> jnp.ndarray:
    """|Σxᵢ²-n|^0.25 + (0.5Σxᵢ²+Σxᵢ)/n + 0.5"""
    D = x.shape[0]
    sum_sq = jnp.sum(x**2)
    sum_x = jnp.sum(x)
    return jnp.abs(sum_sq - D) ** 0.25 + (0.5 * sum_sq + sum_x) / D + 0.5


def _hgbat(x: jnp.ndarray) -> jnp.ndarray:
    """|(Σxᵢ²)²-(Σxᵢ)²|^0.5 + (0.5Σxᵢ²+Σxᵢ)/n + 0.5"""
    D = x.shape[0]
    sum_sq = jnp.sum(x**2)
    sum_x = jnp.sum(x)
    return jnp.abs(sum_sq**2 - sum_x**2) ** 0.5 + (0.5 * sum_sq + sum_x) / D + 0.5


def _whitley(x: jnp.ndarray) -> jnp.ndarray:
    """ΣᵢΣⱼ[yᵢⱼ²/4000 - cos(yᵢⱼ) + 1], yᵢⱼ = 100(xᵢ²-xⱼ)² + (1-xⱼ)²"""
    # Broadcast: x_i -> (D,1), x_j -> (1,D)
    y = 100.0 * (x[:, None] ** 2 - x[None, :]) ** 2 + (1.0 - x[None, :]) ** 2
    return jnp.sum(y**2 / 4000.0 - jnp.cos(y) + 1.0)


def _exponential(x: jnp.ndarray) -> jnp.ndarray:
    """1 - exp(-0.5 Σxᵢ²)  (shifted so f*=0)"""
    return 1.0 - jnp.exp(-0.5 * jnp.sum(x**2))


def _cosine_mixture(x: jnp.ndarray) -> jnp.ndarray:
    """Σ[xᵢ² + 0.1(1 - cos(5πxᵢ))]  (shifted so f*=0)"""
    return jnp.sum(x**2 + 0.1 * (1.0 - jnp.cos(5.0 * jnp.pi * x)))


# ---------------------------------------------------------------------------
# Benchmark registry — sorted by group (unimodal first, then multimodal)
# ---------------------------------------------------------------------------

BENCHMARKS: tuple[BenchmarkFunction, ...] = (
    # --- Unimodal ---
    BenchmarkFunction("Sphere", _sphere, -5.12, 5.12, 0.0, "unimodal"),
    BenchmarkFunction("Rosenbrock", _rosenbrock, -5.0, 10.0, 0.0, "unimodal"),
    BenchmarkFunction("SumSquares", _sum_squares, -10.0, 10.0, 0.0, "unimodal"),
    BenchmarkFunction("Schwefel2.22", _schwefel_2_22, -10.0, 10.0, 0.0, "unimodal"),
    BenchmarkFunction("Schwefel1.2", _schwefel_1_2, -100.0, 100.0, 0.0, "unimodal"),
    BenchmarkFunction("Schwefel2.21", _schwefel_2_21, -100.0, 100.0, 0.0, "unimodal"),
    BenchmarkFunction("Schwefel2.20", _schwefel_2_20, -100.0, 100.0, 0.0, "unimodal"),
    BenchmarkFunction("Schwefel2.23", _schwefel_2_23, -10.0, 10.0, 0.0, "unimodal"),
    BenchmarkFunction("DixonPrice", _dixon_price, -10.0, 10.0, 0.0, "unimodal"),
    BenchmarkFunction("Zakharov", _zakharov, -5.0, 10.0, 0.0, "unimodal"),
    BenchmarkFunction("RotHyperEllipsoid", _rot_hyper_ellipsoid, -65.536, 65.536, 0.0, "unimodal"),
    BenchmarkFunction("SumDiffPowers", _sum_diff_powers, -1.0, 1.0, 0.0, "unimodal"),
    BenchmarkFunction("ChungReynolds", _chung_reynolds, -100.0, 100.0, 0.0, "unimodal"),
    BenchmarkFunction("Quartic", _quartic, -1.28, 1.28, 0.0, "unimodal"),
    BenchmarkFunction("Cigar", _cigar, -100.0, 100.0, 0.0, "unimodal"),
    # --- Multimodal ---
    BenchmarkFunction("Rastrigin", _rastrigin, -5.12, 5.12, 0.0, "multimodal"),
    BenchmarkFunction("Ackley", _ackley, -32.768, 32.768, 0.0, "multimodal"),
    BenchmarkFunction("Griewank", _griewank, -600.0, 600.0, 0.0, "multimodal"),
    BenchmarkFunction("Schwefel", _schwefel, -500.0, 500.0, 0.0, "multimodal"),
    BenchmarkFunction("Levy", _levy, -10.0, 10.0, 0.0, "multimodal"),
    BenchmarkFunction("Bohachevsky", _bohachevsky, -100.0, 100.0, 0.0, "multimodal"),
    BenchmarkFunction("Salomon", _salomon, -100.0, 100.0, 0.0, "multimodal"),
    BenchmarkFunction("Alpine1", _alpine1, -10.0, 10.0, 0.0, "multimodal"),
    BenchmarkFunction("XinSheYang2", _xin_she_yang2, -6.283185307, 6.283185307, 0.0, "multimodal"),
    BenchmarkFunction("Qing", _qing, -500.0, 500.0, 0.0, "multimodal"),
    BenchmarkFunction("Pathological", _pathological, -100.0, 100.0, 0.0, "multimodal"),
    BenchmarkFunction("SchafferF6", _schaffer_f6, -100.0, 100.0, 0.0, "multimodal"),
    BenchmarkFunction("Wavy", _wavy, -3.141592653, 3.141592653, 0.0, "multimodal"),
    BenchmarkFunction("Weierstrass", _weierstrass, -0.5, 0.5, 0.0, "multimodal"),
    BenchmarkFunction("Pinter", _pinter, -10.0, 10.0, 0.0, "multimodal"),
    BenchmarkFunction("StretchedV", _stretched_v, -10.0, 10.0, 0.0, "multimodal"),
    BenchmarkFunction("HappyCat", _happy_cat, -2.0, 2.0, 0.0, "multimodal"),
    BenchmarkFunction("HGBat", _hgbat, -2.0, 2.0, 0.0, "multimodal"),
    BenchmarkFunction("Whitley", _whitley, -10.24, 10.24, 0.0, "multimodal"),
    BenchmarkFunction("Exponential", _exponential, -1.0, 1.0, 0.0, "multimodal"),
    BenchmarkFunction("CosineMixture", _cosine_mixture, -1.0, 1.0, 0.0, "multimodal"),
)
