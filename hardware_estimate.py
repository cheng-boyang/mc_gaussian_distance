"""Estimate throughput, area-normalized throughput, and energy per sample.

This script uses the logical operation-count model from the project together
with simple bandwidth/TDP ceilings for the user's CPU and GPU.

The estimates are roofline-style approximations, not measurements.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from gaussian_distance_cpu import (
    deterministic_operation_counts,
    monte_carlo_operation_counts,
)
from gaussian_distance_gpu import monte_carlo_operation_counts_gpu


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware constants used for first-order estimates."""

    name: str
    bandwidth_gb_s: float
    power_w: float
    area_mm2: float


CPU_W2265 = HardwareSpec(
    name="Intel Xeon W-2265",
    bandwidth_gb_s=93.85,
    power_w=165.0,
    area_mm2=486.0,  # estimate from third-party die-size aggregation
)

GPU_A5000 = HardwareSpec(
    name="NVIDIA RTX A5000",
    bandwidth_gb_s=768.0,
    power_w=230.0,
    area_mm2=628.0,
)
HARDWARE_ESTIMATE_CSV = "hardware_estimate.csv"


def bytes_per_sample(reads: int, writes: int, scalar_bytes: int) -> int:
    """Return total bytes touched per sample."""
    return (reads + writes) * scalar_bytes


def estimate_samples_per_second(spec: HardwareSpec, bytes_each_sample: int) -> float:
    """Memory-bandwidth ceiling for samples per second."""
    return (spec.bandwidth_gb_s * 1e9) / bytes_each_sample


def estimate_energy_per_sample(spec: HardwareSpec, samples_per_second: float) -> float:
    """Return joules per sample from power and throughput."""
    return spec.power_w / samples_per_second


def print_row(method: str, hardware: HardwareSpec, scalar_bytes: int, bytes_sample: int) -> None:
    """Print one comparison row."""
    samples_per_second = estimate_samples_per_second(hardware, bytes_sample)
    samples_per_second_mm2 = samples_per_second / hardware.area_mm2
    joules_per_sample = estimate_energy_per_sample(hardware, samples_per_second)
    samples_per_joule = samples_per_second / hardware.power_w

    print(
        f"{method:<20} | {hardware.name:<18} | {scalar_bytes * 8:>5}-bit | "
        f"{bytes_sample:>12,} | {samples_per_second / 1e6:>12.3f} | "
        f"{samples_per_second_mm2 / 1e6:>12.3f} | {joules_per_sample * 1e9:>12.3f} | "
        f"{samples_per_joule / 1e6:>12.3f}"
    )


def build_row(method: str, hardware: HardwareSpec, scalar_bytes: int, bytes_sample: int) -> dict[str, str | float | int]:
    """Build one CSV row."""
    samples_per_second = estimate_samples_per_second(hardware, bytes_sample)
    samples_per_second_mm2 = samples_per_second / hardware.area_mm2
    joules_per_sample = estimate_energy_per_sample(hardware, samples_per_second)
    samples_per_joule = samples_per_second / hardware.power_w
    return {
        "method": method,
        "hardware": hardware.name,
        "dtype_bits": scalar_bytes * 8,
        "bytes_per_sample": bytes_sample,
        "samples_per_second": samples_per_second,
        "samples_per_second_per_mm2": samples_per_second_mm2,
        "joules_per_sample": joules_per_sample,
        "nanojoules_per_sample": joules_per_sample * 1e9,
        "samples_per_joule": samples_per_joule,
        "mega_samples_per_joule": samples_per_joule / 1e6,
    }


def main() -> None:
    """Print a CPU/GPU comparison table."""
    dim = 3
    quadrature_order = 15
    num_samples = 1_000_000

    det_counts = deterministic_operation_counts(dim, quadrature_order)
    det_bytes_per_eval = bytes_per_sample(
        det_counts.reads // det_counts.sqrts,
        det_counts.writes // det_counts.sqrts,
        scalar_bytes=8,
    )

    mc_counts = monte_carlo_operation_counts(dim, num_samples)
    mc_bytes_per_sample_f64 = bytes_per_sample(
        mc_counts.reads // num_samples,
        mc_counts.writes // num_samples,
        scalar_bytes=8,
    )
    mc_bytes_per_sample_f32 = bytes_per_sample(
        mc_counts.reads // num_samples,
        mc_counts.writes // num_samples,
        scalar_bytes=4,
    )

    gpu_counts = monte_carlo_operation_counts_gpu(dim, num_samples)
    gpu_bytes_per_sample_f64 = bytes_per_sample(
        gpu_counts.reads // num_samples,
        gpu_counts.writes // num_samples,
        scalar_bytes=8,
    )
    gpu_bytes_per_sample_f32 = bytes_per_sample(
        gpu_counts.reads // num_samples,
        gpu_counts.writes // num_samples,
        scalar_bytes=4,
    )

    rows = [
        build_row("CPU deterministic", CPU_W2265, 8, det_bytes_per_eval),
        build_row("CPU Monte Carlo", CPU_W2265, 8, mc_bytes_per_sample_f64),
        build_row("GPU Monte Carlo", GPU_A5000, 4, gpu_bytes_per_sample_f32),
        build_row("GPU Monte Carlo", GPU_A5000, 8, gpu_bytes_per_sample_f64),
    ]

    print("Hardware Estimate Comparison")
    print("memory-bandwidth ceiling, not measured runtime")
    print(
        "method               | hardware           | dtype | bytes/sample |     Msamp/s | Msamp/s/mm2 |    nJ/sample |      MSamp/J"
    )
    print("-" * 132)

    print_row("CPU deterministic", CPU_W2265, 8, det_bytes_per_eval)
    print_row("CPU Monte Carlo", CPU_W2265, 8, mc_bytes_per_sample_f64)
    print_row("GPU Monte Carlo", GPU_A5000, 4, gpu_bytes_per_sample_f32)
    print_row("GPU Monte Carlo", GPU_A5000, 8, gpu_bytes_per_sample_f64)

    with open(HARDWARE_ESTIMATE_CSV, "w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "hardware",
                "dtype_bits",
                "bytes_per_sample",
                "samples_per_second",
                "samples_per_second_per_mm2",
                "joules_per_sample",
                "nanojoules_per_sample",
                "samples_per_joule",
                "mega_samples_per_joule",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {HARDWARE_ESTIMATE_CSV}")


if __name__ == "__main__":
    main()
