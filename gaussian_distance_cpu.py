"""CPU estimators for Gaussian Euclidean distance.

This module estimates

    E[||X - Y||_2]

for independent Gaussian random vectors

    X ~ N(mu1, Sigma1), Y ~ N(mu2, Sigma2).

By independence, the problem reduces to

    Z = X - Y ~ N(mu1 - mu2, Sigma1 + Sigma2),

so the estimator computes E[||Z||_2].
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eigh


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class DistanceResult:
    """Container for estimator outputs."""

    estimate: float
    variance_estimate: float
    runtime_seconds: float
    num_evaluations: int
    dimension: int
    method: str


@dataclass(frozen=True)
class ProfiledDistanceResult(DistanceResult):
    """Estimator result with stage-level timing information."""

    stage_times_seconds: dict[str, float]
    category_times_seconds: dict[str, float]


@dataclass(frozen=True)
class OperationCounts:
    """Estimated operation counts for an estimator run."""

    rng_normals: int
    multiplies: int
    adds: int
    sqrts: int
    reads: int
    writes: int


def _as_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert an input vector to a 1D float64 numpy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    return arr


def _as_2d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert an input matrix to a 2D float64 numpy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr


def validate_gaussian_parameters(
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a Gaussian mean/covariance pair.

    Args:
        mu: Mean vector of shape (d,).
        sigma: Covariance matrix of shape (d, d).
        atol: Absolute tolerance for symmetry and PSD checks.

    Returns:
        A tuple ``(mu_array, sigma_array)`` in float64 format.

    Raises:
        ValueError: If shapes are inconsistent, the covariance is not symmetric,
            or the covariance is not positive semidefinite within tolerance.
    """
    mu_arr = _as_1d_array(mu, "mu")
    sigma_arr = _as_2d_array(sigma, "sigma")

    d = mu_arr.shape[0]
    if sigma_arr.shape != (d, d):
        raise ValueError(
            f"sigma must have shape {(d, d)} to match mu, got {sigma_arr.shape}."
        )

    if not np.allclose(sigma_arr, sigma_arr.T, atol=atol, rtol=0.0):
        raise ValueError("sigma must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(sigma_arr)
    min_eig = float(np.min(eigenvalues))
    if min_eig < -atol:
        raise ValueError(
            f"sigma must be positive semidefinite; minimum eigenvalue is {min_eig:.3e}."
        )

    return mu_arr, sigma_arr


def covariance_matrix_sqrt(sigma: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Compute a symmetric square root of a PSD covariance matrix.

    Uses an eigenvalue decomposition and clamps tiny negative eigenvalues caused
    by numerical roundoff to zero.
    """
    eigenvalues, eigenvectors = eigh(sigma, check_finite=True)
    eigenvalues = np.where(eigenvalues < 0.0, np.maximum(eigenvalues, -atol), eigenvalues)
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    sqrt_diag = np.sqrt(eigenvalues)
    return (eigenvectors * sqrt_diag) @ eigenvectors.T


def reduced_gaussian_parameters(
    mu1: ArrayLike,
    sigma1: ArrayLike,
    mu2: ArrayLike,
    sigma2: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the reduced Gaussian Z = X - Y."""
    mu1_arr, sigma1_arr = validate_gaussian_parameters(mu1, sigma1)
    mu2_arr, sigma2_arr = validate_gaussian_parameters(mu2, sigma2)

    if mu1_arr.shape != mu2_arr.shape:
        raise ValueError(
            f"mu1 and mu2 must have the same shape, got {mu1_arr.shape} and {mu2_arr.shape}."
        )

    return mu1_arr - mu2_arr, sigma1_arr + sigma2_arr


def sample_gaussian_norms_cpu(
    mu: ArrayLike,
    sigma: ArrayLike,
    num_samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample Euclidean norms from a Gaussian random vector on CPU.

    Args:
        mu: Mean vector of shape (d,).
        sigma: Covariance matrix of shape (d, d).
        num_samples: Number of Monte Carlo samples.
        rng: Optional numpy random generator.

    Returns:
        A vector of shape (num_samples,) containing sampled norms.
    """
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1.")

    mu_arr, sigma_arr = validate_gaussian_parameters(mu, sigma)
    rng = np.random.default_rng() if rng is None else rng

    sqrt_sigma = covariance_matrix_sqrt(sigma_arr)
    standard = rng.standard_normal(size=(num_samples, mu_arr.shape[0]))
    samples = mu_arr + standard @ sqrt_sigma.T
    return np.linalg.norm(samples, axis=1)


def deterministic_expected_norm_cpu(
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    quadrature_order: int = 15,
) -> DistanceResult:
    """Estimate E[||Z||_2] with deterministic Gauss-Hermite quadrature on CPU.

    This method is non-Monte-Carlo and is practical for modest dimensions.
    For dimension ``d`` and quadrature order ``m``, it evaluates the integrand
    on ``m ** d`` grid points.
    """
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    start = perf_counter()
    mu_arr, sigma_arr = validate_gaussian_parameters(mu, sigma)
    sqrt_sigma = covariance_matrix_sqrt(sigma_arr)
    dim = mu_arr.shape[0]

    nodes_1d, weights_1d = hermgauss(quadrature_order)
    mesh = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
    grid = np.stack([axis.reshape(-1) for axis in mesh], axis=1)

    weight_mesh = np.meshgrid(*([weights_1d] * dim), indexing="ij")
    weights = np.prod(np.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)

    transformed = mu_arr + (np.sqrt(2.0) * grid) @ sqrt_sigma.T
    norms = np.linalg.norm(transformed, axis=1)
    normalization = np.pi ** (-0.5 * dim)
    estimate = normalization * np.dot(weights, norms)
    runtime = perf_counter() - start

    return DistanceResult(
        estimate=float(estimate),
        variance_estimate=0.0,
        runtime_seconds=runtime,
        num_evaluations=int(quadrature_order**dim),
        dimension=int(dim),
        method="cpu_deterministic",
    )


def monte_carlo_expected_norm_cpu(
    mu: ArrayLike,
    sigma: ArrayLike,
    num_samples: int,
    *,
    seed: Optional[int] = None,
) -> DistanceResult:
    """Estimate E[||Z||_2] for a Gaussian random vector on CPU.

    The reported variance is the Monte Carlo estimator variance, i.e.

        Var(mean(norms)) ~= sample_var(norms) / num_samples.
    """
    start = perf_counter()
    rng = np.random.default_rng(seed)
    norms = sample_gaussian_norms_cpu(mu, sigma, num_samples, rng=rng)
    sample_var = float(np.var(norms, ddof=1))
    runtime = perf_counter() - start
    return DistanceResult(
        estimate=float(np.mean(norms)),
        variance_estimate=sample_var / num_samples,
        runtime_seconds=runtime,
        num_evaluations=num_samples,
        dimension=int(np.asarray(mu).shape[0]),
        method="cpu_monte_carlo",
    )


def deterministic_expected_distance_cpu(
    mu1: ArrayLike,
    sigma1: ArrayLike,
    mu2: ArrayLike,
    sigma2: ArrayLike,
    *,
    quadrature_order: int = 15,
) -> DistanceResult:
    """Estimate E[||X - Y||_2] on CPU with deterministic quadrature."""
    mu_z, sigma_z = reduced_gaussian_parameters(mu1, sigma1, mu2, sigma2)
    return deterministic_expected_norm_cpu(mu_z, sigma_z, quadrature_order=quadrature_order)


def monte_carlo_expected_distance_cpu(
    mu1: ArrayLike,
    sigma1: ArrayLike,
    mu2: ArrayLike,
    sigma2: ArrayLike,
    num_samples: int,
    *,
    seed: Optional[int] = None,
) -> DistanceResult:
    """Estimate E[||X - Y||_2] for independent Gaussians on CPU."""
    mu_z, sigma_z = reduced_gaussian_parameters(mu1, sigma1, mu2, sigma2)
    return monte_carlo_expected_norm_cpu(mu_z, sigma_z, num_samples, seed=seed)


def profile_monte_carlo_expected_norm_cpu(
    mu: ArrayLike,
    sigma: ArrayLike,
    num_samples: int,
    *,
    seed: Optional[int] = None,
) -> ProfiledDistanceResult:
    """Profile CPU Monte Carlo by separating setup, RNG, compute, and reduction."""
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1.")

    total_start = perf_counter()
    mu_arr, sigma_arr = validate_gaussian_parameters(mu, sigma)

    t0 = perf_counter()
    sqrt_sigma = covariance_matrix_sqrt(sigma_arr)
    rng = np.random.default_rng(seed)
    t1 = perf_counter()

    standard = rng.standard_normal(size=(num_samples, mu_arr.shape[0]))
    t2 = perf_counter()

    samples = mu_arr + standard @ sqrt_sigma.T
    norms = np.linalg.norm(samples, axis=1)
    t3 = perf_counter()

    estimate = float(np.mean(norms))
    sample_var = float(np.var(norms, ddof=1))
    t4 = perf_counter()

    return ProfiledDistanceResult(
        estimate=estimate,
        variance_estimate=sample_var / num_samples,
        runtime_seconds=t4 - total_start,
        num_evaluations=num_samples,
        dimension=int(mu_arr.shape[0]),
        method="cpu_monte_carlo_profiled",
        stage_times_seconds={
            "setup": t1 - t0,
            "rng": t2 - t1,
            "transform_norm": t3 - t2,
            "reduction": t4 - t3,
        },
        category_times_seconds={
            "rng": t2 - t1,
            "mem_access": t1 - t0,
            "compute": (t3 - t2) + (t4 - t3),
        },
    )


def profile_monte_carlo_expected_distance_cpu(
    mu1: ArrayLike,
    sigma1: ArrayLike,
    mu2: ArrayLike,
    sigma2: ArrayLike,
    num_samples: int,
    *,
    seed: Optional[int] = None,
) -> ProfiledDistanceResult:
    """Profile E[||X - Y||_2] CPU Monte Carlo stage timings."""
    mu_z, sigma_z = reduced_gaussian_parameters(mu1, sigma1, mu2, sigma2)
    return profile_monte_carlo_expected_norm_cpu(mu_z, sigma_z, num_samples, seed=seed)


def deterministic_operation_counts(dimension: int, quadrature_order: int) -> OperationCounts:
    """Return estimated operation counts for deterministic quadrature."""
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive.")

    evaluations = quadrature_order**dimension
    multiplies_per_eval = dimension + dimension * dimension + dimension + 1
    adds_per_eval = dimension * (dimension - 1) + (dimension - 1) + 1 + dimension
    sqrts_per_eval = 1
    reads_per_eval = (dimension * dimension) + (3 * dimension) + 1
    writes_per_eval = 2 * dimension + 1

    return OperationCounts(
        rng_normals=0,
        multiplies=evaluations * multiplies_per_eval,
        adds=evaluations * adds_per_eval,
        sqrts=evaluations * sqrts_per_eval,
        reads=evaluations * reads_per_eval,
        writes=evaluations * writes_per_eval,
    )


def monte_carlo_operation_counts(dimension: int, num_samples: int) -> OperationCounts:
    """Return estimated operation counts for Monte Carlo in dimension d."""
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    multiplies_per_sample = (dimension * dimension) + dimension + 1
    adds_per_sample = dimension * (dimension - 1) + dimension + (dimension - 1)
    reads_per_sample = (dimension * dimension) + (3 * dimension) + 1
    writes_per_sample = (2 * dimension) + 1

    reduction_multiplies = num_samples
    reduction_adds = 2 * max(0, num_samples - 1)
    reduction_reads = 2 * num_samples

    return OperationCounts(
        rng_normals=dimension * num_samples,
        multiplies=(num_samples * multiplies_per_sample) + reduction_multiplies,
        adds=(num_samples * adds_per_sample) + reduction_adds,
        sqrts=num_samples,
        reads=(num_samples * reads_per_sample) + reduction_reads,
        writes=num_samples * writes_per_sample,
    )
