"""Hardware-oriented timing breakdown for Gaussian distance estimators.

This script focuses on:
- CPU deterministic quadrature
- CPU Monte Carlo with stage timings
- GPU Monte Carlo with stage timings

It reports cold vs warm GPU timings separately because first-use CUDA overhead
is large relative to a 3D workload.
"""

from __future__ import annotations

import csv
from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from benchmark import get_test_cases
from gaussian_distance_cpu import (
    deterministic_expected_distance_cpu,
    deterministic_operation_counts,
    monte_carlo_operation_counts,
    profile_monte_carlo_expected_distance_cpu,
)
from gaussian_distance_gpu import (
    monte_carlo_operation_counts_gpu,
    profile_monte_carlo_expected_distance_gpu,
)


SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
BATCH_SIZES = [10_000, 50_000, 250_000, 1_000_000]
RESULTS_DIR = "results_csv"
OPERATION_BREAKDOWN_CSV = f"{RESULTS_DIR}/operation_breakdown.csv"
GPU_SWEEP_CSV = f"{RESULTS_DIR}/gpu_operation_sweep.csv"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:9.3f}"


def _fmt_int(value: int) -> str:
    return f"{value:>12,}"


def _fmt_rate(num_samples: int, seconds: float) -> str:
    if seconds <= 0.0:
        return "      inf"
    return f"{num_samples / seconds / 1e6:9.3f}"


def _estimate_gpu_bytes_per_sample(dtype_bytes: int, dim: int = 3) -> int:
    """Rough global-memory traffic estimate per sample in bytes.

    This is intentionally approximate and is meant for hardware reasoning:
    - RNG output tensor: dim values written
    - transform input/output traffic
    - norm input/output traffic
    - reduction reads
    """
    return dtype_bytes * (dim + 2 * dim + dim + 1)


def _fmt_bandwidth(num_samples: int, seconds: float, dtype_bytes: int, dim: int = 3) -> str:
    if seconds <= 0.0:
        return "      inf"
    total_bytes = num_samples * _estimate_gpu_bytes_per_sample(dtype_bytes, dim=dim)
    gib_per_sec = total_bytes / seconds / (1024**3)
    return f"{gib_per_sec:9.3f}"


def _print_cpu_det(case_name: str, params: dict) -> None:
    result = deterministic_expected_distance_cpu(**params, quadrature_order=15)
    counts = deterministic_operation_counts(dimension=3, quadrature_order=15)
    print(
        f"{case_name:<8} | {'CPU deterministic':<18} | "
        f"{result.num_evaluations:>9,d} | {_fmt_int(counts.rng_normals)} | "
        f"{_fmt_int(counts.multiplies)} | {_fmt_int(counts.adds)} | "
        f"{_fmt_int(counts.sqrts)} | {_fmt_int(counts.reads)} | {_fmt_int(counts.writes)}"
    )


def _print_count_row(case_name: str, label: str, count: int, counts) -> None:
    print(
        f"{case_name:<8} | {label:<18} | {count:>9,d} | {_fmt_int(counts.rng_normals)} | "
        f"{_fmt_int(counts.multiplies)} | {_fmt_int(counts.adds)} | {_fmt_int(counts.sqrts)} | "
        f"{_fmt_int(counts.reads)} | {_fmt_int(counts.writes)}"
    )


def _csv_row(case_name: str, method: str, count: int, counts) -> dict[str, int | str]:
    return {
        "case": case_name,
        "method": method,
        "count": count,
        "rng_normals": counts.rng_normals,
        "multiplies": counts.multiplies,
        "adds": counts.adds,
        "sqrts": counts.sqrts,
        "reads": counts.reads,
        "writes": counts.writes,
    }


def run_profile(sample_sizes: Iterable[int] = SAMPLE_SIZES, seed: int = 1234) -> None:
    """Print estimated operation counts for the three methods."""
    rows: list[dict[str, int | str]] = []
    print("Gaussian Distance Operation Breakdown")
    print(
        "case     | method             |     count |  rng_normals |   multiplies |         adds |        sqrts |        reads |       writes"
    )
    print("-" * 140)

    has_cuda = torch is not None and torch.cuda.is_available()
    for case_name, params in get_test_cases().items():
        _print_cpu_det(case_name, params)
        rows.append(
            _csv_row(
                case_name,
                "CPU deterministic",
                15**3,
                deterministic_operation_counts(dimension=3, quadrature_order=15),
            )
        )
        for n in sample_sizes:
            cpu_counts = monte_carlo_operation_counts(dimension=3, num_samples=n)
            _print_count_row(case_name, "CPU Monte Carlo", n, cpu_counts)
            rows.append(_csv_row(case_name, "CPU Monte Carlo", n, cpu_counts))

            if has_cuda:
                gpu_counts = monte_carlo_operation_counts_gpu(dimension=3, num_samples=n)
                _print_count_row(case_name, "GPU Monte Carlo", n, gpu_counts)
                rows.append(_csv_row(case_name, "GPU Monte Carlo", n, gpu_counts))
            else:
                print(
                    f"{case_name:<8} | {'GPU Monte Carlo':<18} | {n:>9,d} | "
                    f"{'CUDA off':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>12}"
                )
                rows.append(
                    {
                        "case": case_name,
                        "method": "GPU Monte Carlo",
                        "count": n,
                        "rng_normals": "CUDA off",
                        "multiplies": "N/A",
                        "adds": "N/A",
                        "sqrts": "N/A",
                        "reads": "N/A",
                        "writes": "N/A",
                    }
                )

    with open(OPERATION_BREAKDOWN_CSV, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "method",
                "count",
                "rng_normals",
                "multiplies",
                "adds",
                "sqrts",
                "reads",
                "writes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OPERATION_BREAKDOWN_CSV}")


def run_gpu_sweep(case_name: str = "case_1", num_samples: int = 1_000_000, seed: int = 1234) -> None:
    """Run an A5000-oriented GPU sweep over dtype and batch size.

    This reports:
    - cold vs warm runtime
    - stage-level timings
    - samples/sec
    - rough effective bandwidth estimate
    """
    del seed
    print(f"\nGPU Sweep for {case_name}, samples={num_samples:,}")
    print("estimated logical operation counts; counts do not depend on dtype or batch size")
    print("dtype     | batch_size |  rng_normals |   multiplies |         adds |        sqrts |        reads |       writes")
    print("-" * 122)

    rows: list[dict[str, int | str]] = []
    sweep = ["float32", "float64"]
    counts = monte_carlo_operation_counts_gpu(dimension=3, num_samples=num_samples)

    for dtype_name in sweep:
        for batch_size in BATCH_SIZES:
            current_batch = min(batch_size, num_samples)
            print(
                f"{dtype_name:<9} | {current_batch:>10,d} | {_fmt_int(counts.rng_normals)} | "
                f"{_fmt_int(counts.multiplies)} | {_fmt_int(counts.adds)} | {_fmt_int(counts.sqrts)} | "
                f"{_fmt_int(counts.reads)} | {_fmt_int(counts.writes)}"
            )
            rows.append(
                {
                    "case": case_name,
                    "dtype": dtype_name,
                    "batch_size": current_batch,
                    "count": num_samples,
                    "rng_normals": counts.rng_normals,
                    "multiplies": counts.multiplies,
                    "adds": counts.adds,
                    "sqrts": counts.sqrts,
                    "reads": counts.reads,
                    "writes": counts.writes,
                }
            )

    with open(GPU_SWEEP_CSV, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "dtype",
                "batch_size",
                "count",
                "rng_normals",
                "multiplies",
                "adds",
                "sqrts",
                "reads",
                "writes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {GPU_SWEEP_CSV}")


if __name__ == "__main__":
    run_profile()
    run_gpu_sweep()
