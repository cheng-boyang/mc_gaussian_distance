"""Benchmark deterministic CPU, CPU Monte Carlo, and GPU Monte Carlo estimators."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None

from gaussian_distance_cpu import (
    deterministic_expected_distance_cpu,
    monte_carlo_expected_distance_cpu,
)
from gaussian_distance_gpu import monte_carlo_expected_distance_gpu


SAMPLE_SIZES = [10_000, 100_000, 1_000_000]


def get_test_cases():
    """Return five benchmark cases."""
    return {
        "case_1": {
            "mu1": np.array([0.0, 0.0, 0.0]),
            "mu2": np.array([1.0, 2.0, 3.0]),
            "sigma1": np.eye(3),
            "sigma2": np.eye(3),
        },
        "case_2": {
            "mu1": np.array([0.0, 0.0, 0.0]),
            "mu2": np.array([1.0, 1.0, 1.0]),
            "sigma1": np.diag([1.0, 2.0, 3.0]),
            "sigma2": np.diag([2.0, 1.0, 0.5]),
        },
        "case_3": {
            "mu1": np.array([0.0, 0.0, 0.0]),
            "mu2": np.array([0.5, -1.5, 2.0]),
            "sigma1": 0.25 * np.eye(3),
            "sigma2": 4.0 * np.eye(3),
        },
        "case_4": {
            "mu1": np.array([1.0, -1.0, 0.5]),
            "mu2": np.array([-0.5, 0.5, 1.0]),
            "sigma1": np.array(
                [[1.0, 0.9, 0.0], [0.9, 0.81, 0.0], [0.0, 0.0, 0.2]]
            ),
            "sigma2": np.diag([0.1, 0.2, 0.3]),
        },
        "case_5": {
            "mu1": np.array([2.0, -1.0, 0.5]),
            "mu2": np.array([-1.0, 3.0, -2.5]),
            "sigma1": np.zeros((3, 3)),
            "sigma2": np.zeros((3, 3)),
        },
    }


def _format_result(case_name: str, method: str, count_label: str, estimate: float, variance: float, runtime: float) -> str:
    stderr = variance ** 0.5
    return (
        f"{case_name:<8} | {method:<16} | {count_label:>9} | "
        f"{estimate:>12.6f} | {stderr:>12.6e} | {runtime:>10.4f}"
    )


def run_benchmark(sample_sizes: Iterable[int] = SAMPLE_SIZES, seed: int = 1234) -> None:
    """Print a clean timing table comparing the three estimators."""
    print("Gaussian Distance Benchmark")
    print("case     | method           | evaluations |     estimate |       stderr |  time (s)")
    print("-" * 89)

    has_cuda = torch is not None and torch.cuda.is_available()
    for case_name, params in get_test_cases().items():
        det_result = deterministic_expected_distance_cpu(**params, quadrature_order=15)
        print(
            _format_result(
                case_name,
                "CPU deterministic",
                f"{det_result.num_evaluations:,}",
                det_result.estimate,
                det_result.variance_estimate,
                det_result.runtime_seconds,
            )
        )

        for n in sample_sizes:
            cpu_result = monte_carlo_expected_distance_cpu(**params, num_samples=n, seed=seed)
            print(
                _format_result(
                    case_name,
                    "CPU Monte Carlo",
                    f"{n:,}",
                    cpu_result.estimate,
                    cpu_result.variance_estimate,
                    cpu_result.runtime_seconds,
                )
            )

            if has_cuda:
                gpu_result = monte_carlo_expected_distance_gpu(
                    **params,
                    num_samples=n,
                    seed=seed,
                    dtype=torch.float64,
                    batch_size=min(n, 250_000),
                )
                print(
                    _format_result(
                        case_name,
                        "GPU Monte Carlo",
                        f"{n:,}",
                        gpu_result.estimate,
                        gpu_result.variance_estimate,
                        gpu_result.runtime_seconds,
                    )
                )
            else:
                print(
                    f"{case_name:<8} | {'GPU Monte Carlo':<16} | {f'{n:,}':>9} | "
                    f"{'N/A':>12} | {'N/A':>12} | {'CUDA off':>10}"
                )


if __name__ == "__main__":
    run_benchmark()
