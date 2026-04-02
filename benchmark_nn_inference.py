"""Benchmark CPU vs GPU inference for the NN Gaussian sampler.

This script:
1. Trains the two-layer ReLU surrogate once.
2. Copies the trained weights to CPU and GPU inference models.
3. Benchmarks Monte Carlo distance estimation throughput on CPU and GPU.

The benchmark focuses on NN inference only. Training is not included in the
reported inference timings.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from nn_gaussian_distance import (
    TwoLayerGaussianSampler,
    get_device,
    monte_carlo_distance_nn,
    parse_vec3,
    train_surrogate,
)


DEFAULT_SAMPLE_SIZES = [10_000, 100_000, 1_000_000]
DEFAULT_OUTPUT_CSV = "results_csv/nn_inference_benchmark.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the NN inference benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mu1", default="0,0,0")
    parser.add_argument("--std1", default="1,1,1")
    parser.add_argument("--mu2", default="1,2,3")
    parser.add_argument("--std2", default="1,1,1")
    parser.add_argument("--sample-sizes", default="10000,100000,1000000")
    parser.add_argument("--cpu-batch-size", type=int, default=50000)
    parser.add_argument("--gpu-batch-size", type=int, default=250000)
    parser.add_argument("--warmup-samples", type=int, default=50000)
    parser.add_argument("--csv", default=DEFAULT_OUTPUT_CSV)
    return parser


def parse_sample_sizes(text: str) -> list[int]:
    """Parse comma-separated sample sizes."""
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one sample size is required.")
    return values


def clone_model_to_device(model: TwoLayerGaussianSampler, device: torch.device) -> TwoLayerGaussianSampler:
    """Clone model weights onto a target device."""
    first = model.net[0].out_features
    second = model.net[2].out_features
    clone = TwoLayerGaussianSampler(first, second).to(device)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    return clone


def make_case_tensors(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Build case tensors on a target device."""
    mu1 = parse_vec3(args.mu1).to(device).unsqueeze(0)
    std1 = parse_vec3(args.std1).to(device).unsqueeze(0)
    mu2 = parse_vec3(args.mu2).to(device).unsqueeze(0)
    std2 = parse_vec3(args.std2).to(device).unsqueeze(0)
    return mu1, std1, mu2, std2


def benchmark_one(
    label: str,
    model: TwoLayerGaussianSampler,
    device: torch.device,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    *,
    num_samples: int,
    batch_size: int,
) -> dict[str, float | int | str]:
    """Benchmark one inference run."""
    summary = monte_carlo_distance_nn(
        model,
        mu1,
        std1,
        mu2,
        std2,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
    )
    samples_per_second = num_samples / summary.runtime_seconds
    return {
        "device": label,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "estimate": summary.estimate,
        "stderr": summary.variance_estimate**0.5,
        "runtime_seconds": summary.runtime_seconds,
        "samples_per_second": samples_per_second,
        "million_samples_per_second": samples_per_second / 1e6,
    }


def print_row(row: dict[str, float | int | str]) -> None:
    """Print one benchmark row."""
    print(
        f"{str(row['device']):<8} | {int(row['num_samples']):>10,d} | {int(row['batch_size']):>10,d} | "
        f"{float(row['estimate']):>10.6f} | {float(row['stderr']):>10.6e} | "
        f"{float(row['runtime_seconds']):>10.4f} | {float(row['million_samples_per_second']):>10.3f}"
    )


def main() -> None:
    """Train once and benchmark CPU vs GPU NN inference."""
    args = build_parser().parse_args()
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    torch.manual_seed(args.seed)

    train_device = get_device("cpu")
    train_model = TwoLayerGaussianSampler(args.hidden1, args.hidden2).to(train_device)
    print("training NN surrogate on CPU...")
    train_surrogate(
        train_model,
        train_steps=args.train_steps,
        batch_size=args.train_batch_size,
        lr=args.lr,
        device=train_device,
    )

    cpu_device = torch.device("cpu")
    cpu_model = clone_model_to_device(train_model, cpu_device)
    cpu_case = make_case_tensors(args, cpu_device)

    has_cuda = torch.cuda.is_available()
    gpu_model = None
    gpu_case = None
    gpu_device = None
    if has_cuda:
        gpu_device = torch.device("cuda")
        gpu_model = clone_model_to_device(train_model, gpu_device)
        gpu_case = make_case_tensors(args, gpu_device)
        monte_carlo_distance_nn(
            gpu_model,
            *gpu_case,
            num_samples=args.warmup_samples,
            batch_size=min(args.gpu_batch_size, args.warmup_samples),
            device=gpu_device,
        )

    rows: list[dict[str, float | int | str]] = []

    print("\nNN inference benchmark")
    print("device   |    samples | batch_size |   estimate |     stderr |   time (s) |   Msamp/s")
    print("-" * 88)
    for n in sample_sizes:
        cpu_row = benchmark_one(
            "CPU",
            cpu_model,
            cpu_device,
            *cpu_case,
            num_samples=n,
            batch_size=min(args.cpu_batch_size, n),
        )
        print_row(cpu_row)
        rows.append(cpu_row)

        if has_cuda and gpu_model is not None and gpu_case is not None and gpu_device is not None:
            gpu_row = benchmark_one(
                "GPU",
                gpu_model,
                gpu_device,
                *gpu_case,
                num_samples=n,
                batch_size=min(args.gpu_batch_size, n),
            )
            print_row(gpu_row)
            rows.append(gpu_row)
        else:
            print(
                f"{'GPU':<8} | {n:>10,d} | {min(args.gpu_batch_size, n):>10,d} | "
                f"{'N/A':>10} | {'N/A':>10} | {'CUDA off':>10} | {'N/A':>10}"
            )
            rows.append(
                {
                    "device": "GPU",
                    "num_samples": n,
                    "batch_size": min(args.gpu_batch_size, n),
                    "estimate": "N/A",
                    "stderr": "N/A",
                    "runtime_seconds": "CUDA off",
                    "samples_per_second": "N/A",
                    "million_samples_per_second": "N/A",
                }
            )

    output_path = Path(args.csv)
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device",
                "num_samples",
                "batch_size",
                "estimate",
                "stderr",
                "runtime_seconds",
                "samples_per_second",
                "million_samples_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
