"""GPU Monte Carlo estimators for Gaussian Euclidean distance."""

from __future__ import annotations

from time import perf_counter
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None

from gaussian_distance_cpu import (
    DistanceResult,
    OperationCounts,
    ProfiledDistanceResult,
    covariance_matrix_sqrt,
    monte_carlo_operation_counts,
    reduced_gaussian_parameters,
    validate_gaussian_parameters,
)


def get_default_device() -> torch.device:
    """Return a CUDA device when available."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch with CUDA support.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch build.")
    return torch.device("cuda")


def monte_carlo_expected_norm_gpu(
    mu,
    sigma,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    batch_size: Optional[int] = None,
) -> DistanceResult:
    """Estimate E[||Z||_2] for a Gaussian random vector on GPU.

    Args:
        mu: Mean vector of shape (d,).
        sigma: Covariance matrix of shape (d, d).
        num_samples: Number of Monte Carlo samples.
        seed: Optional random seed.
        device: Target torch device. Defaults to the current CUDA device.
        dtype: Floating-point dtype used on device.
        batch_size: Optional number of samples per batch. Defaults to all samples.

    Returns:
        DistanceResult with estimate, estimator variance, and runtime.
    """
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1.")
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch with CUDA support.")

    mu_arr, sigma_arr = validate_gaussian_parameters(mu, sigma)
    sigma_sqrt = covariance_matrix_sqrt(sigma_arr)

    device = get_default_device() if device is None else device
    batch_size = num_samples if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    mu_t = torch.as_tensor(mu_arr, dtype=dtype, device=device)
    sigma_sqrt_t = torch.as_tensor(sigma_sqrt, dtype=dtype, device=device)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    start = perf_counter()
    sum_norms = torch.zeros((), dtype=dtype, device=device)
    sum_sq_norms = torch.zeros((), dtype=dtype, device=device)

    remaining = num_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        eps = torch.randn(
            (current_batch, mu_arr.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        samples = mu_t + eps @ sigma_sqrt_t.T
        norms = torch.linalg.norm(samples, dim=1)
        sum_norms += norms.sum()
        sum_sq_norms += (norms * norms).sum()
        remaining -= current_batch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = perf_counter() - start

    n = float(num_samples)
    mean = sum_norms / n
    second_moment = sum_sq_norms / n
    sample_var = (n / (n - 1.0)) * (second_moment - mean * mean)
    sample_var = torch.clamp(sample_var, min=0.0)
    estimator_var = sample_var / n

    return DistanceResult(
        estimate=float(mean.item()),
        variance_estimate=float(estimator_var.item()),
        runtime_seconds=runtime,
        num_evaluations=num_samples,
        dimension=int(mu_arr.shape[0]),
        method="gpu_monte_carlo",
    )


def monte_carlo_expected_distance_gpu(
    mu1,
    sigma1,
    mu2,
    sigma2,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    batch_size: Optional[int] = None,
) -> DistanceResult:
    """Estimate E[||X - Y||_2] for independent Gaussians on GPU."""
    mu_z, sigma_z = reduced_gaussian_parameters(mu1, sigma1, mu2, sigma2)
    return monte_carlo_expected_norm_gpu(
        mu_z,
        sigma_z,
        num_samples,
        seed=seed,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )


def profile_monte_carlo_expected_norm_gpu(
    mu,
    sigma,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    batch_size: Optional[int] = None,
) -> ProfiledDistanceResult:
    """Profile GPU Monte Carlo by separating setup, RNG, compute, and reduction."""
    if num_samples <= 1:
        raise ValueError("num_samples must be greater than 1.")
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch with CUDA support.")

    total_start = perf_counter()
    mu_arr, sigma_arr = validate_gaussian_parameters(mu, sigma)

    t0 = perf_counter()
    sigma_sqrt = covariance_matrix_sqrt(sigma_arr)
    device = get_default_device() if device is None else device
    batch_size = num_samples if batch_size is None else int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    mu_t = torch.as_tensor(mu_arr, dtype=dtype, device=device)
    sigma_sqrt_t = torch.as_tensor(sigma_sqrt, dtype=dtype, device=device)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = perf_counter()

    rng_time = 0.0
    transform_norm_time = 0.0
    reduction_time = 0.0
    sum_norms = torch.zeros((), dtype=dtype, device=device)
    sum_sq_norms = torch.zeros((), dtype=dtype, device=device)

    remaining = num_samples
    while remaining > 0:
        current_batch = min(batch_size, remaining)

        b0 = perf_counter()
        eps = torch.randn(
            (current_batch, mu_arr.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        b1 = perf_counter()

        samples = mu_t + eps @ sigma_sqrt_t.T
        norms = torch.linalg.norm(samples, dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        b2 = perf_counter()

        sum_norms += norms.sum()
        sum_sq_norms += (norms * norms).sum()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        b3 = perf_counter()

        rng_time += b1 - b0
        transform_norm_time += b2 - b1
        reduction_time += b3 - b2
        remaining -= current_batch

    n = float(num_samples)
    mean = sum_norms / n
    second_moment = sum_sq_norms / n
    sample_var = (n / (n - 1.0)) * (second_moment - mean * mean)
    sample_var = torch.clamp(sample_var, min=0.0)
    estimator_var = sample_var / n
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t2 = perf_counter()

    return ProfiledDistanceResult(
        estimate=float(mean.item()),
        variance_estimate=float(estimator_var.item()),
        runtime_seconds=t2 - total_start,
        num_evaluations=num_samples,
        dimension=int(mu_arr.shape[0]),
        method="gpu_monte_carlo_profiled",
        stage_times_seconds={
            "setup": t1 - t0,
            "rng": rng_time,
            "transform_norm": transform_norm_time,
            "reduction": reduction_time,
            "finalize": t2 - t1 - rng_time - transform_norm_time - reduction_time,
        },
        category_times_seconds={
            "rng": rng_time,
            "mem_access": (t1 - t0) + max(
                0.0, t2 - t1 - rng_time - transform_norm_time - reduction_time
            ),
            "compute": transform_norm_time + reduction_time,
        },
    )


def profile_monte_carlo_expected_distance_gpu(
    mu1,
    sigma1,
    mu2,
    sigma2,
    num_samples: int,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    batch_size: Optional[int] = None,
) -> ProfiledDistanceResult:
    """Profile E[||X - Y||_2] GPU Monte Carlo stage timings."""
    mu_z, sigma_z = reduced_gaussian_parameters(mu1, sigma1, mu2, sigma2)
    return profile_monte_carlo_expected_norm_gpu(
        mu_z,
        sigma_z,
        num_samples,
        seed=seed,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )


def monte_carlo_operation_counts_gpu(dimension: int, num_samples: int) -> OperationCounts:
    """Return estimated GPU Monte Carlo operation counts.

    The arithmetic count model matches the CPU Monte Carlo algorithm. Real GPU
    execution may fuse kernels or use different reduction trees, but the logical
    operation count is the same.
    """
    return monte_carlo_operation_counts(dimension, num_samples)
