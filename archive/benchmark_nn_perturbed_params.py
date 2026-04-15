"""Benchmark GPU latency for NN inference with perturbed Gaussian parameters.

Flow per batch k:
1. Sample a batch of perturbed parameters from Gaussian distributions.
2. Sample eps ~ N(0, I_3).
3. Feed parameters and eps into the unchanged NN surrogate.
4. Compute Euclidean norms and a batch mean.

The default benchmark uses K=20 parameter batches, as requested.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch

from nn_gaussian_distance import (
    TwoLayerGaussianSampler,
    get_device,
    parse_vec3,
    train_surrogate,
)


DEFAULT_CSV = "results_csv/nn_perturbed_params_latency.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-batches", type=int, default=20)
    parser.add_argument("--param-batch-size", type=int, default=50000)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--mu1", default="0,0,0")
    parser.add_argument("--std1", default="1,1,1")
    parser.add_argument("--mu2", default="1,2,3")
    parser.add_argument("--std2", default="1,1,1")
    parser.add_argument("--mu1-perturb-std", default="0.1,0.1,0.1")
    parser.add_argument("--std1-perturb-std", default="0.05,0.05,0.05")
    parser.add_argument("--mu2-perturb-std", default="0.1,0.1,0.1")
    parser.add_argument("--std2-perturb-std", default="0.05,0.05,0.05")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument(
        "--save-model",
        default=None,
        help="Path to save the trained model checkpoint (.pt) after training.",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        help="Path to load a model checkpoint (.pt), skipping training entirely.",
    )
    return parser


def clone_model_to_device(model: TwoLayerGaussianSampler, device: torch.device) -> TwoLayerGaussianSampler:
    """Clone trained weights to a target device."""
    clone = TwoLayerGaussianSampler(model.net[0].out_features, model.net[2].out_features).to(device)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    return clone


def sample_parameter_batch(
    center: torch.Tensor,
    perturb_std: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Sample a perturbed parameter batch."""
    return center + perturb_std * torch.randn(batch_size, 3, device=center.device)


@torch.no_grad()
def run_latency_benchmark(
    model: TwoLayerGaussianSampler,
    *,
    k_batches: int,
    param_batch_size: int,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    mu1_perturb_std: torch.Tensor,
    std1_perturb_std: torch.Tensor,
    mu2_perturb_std: torch.Tensor,
    std2_perturb_std: torch.Tensor,
    warmup_batches: int,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """Run warmup plus measured K-batch GPU latency benchmark."""
    device = mu1.device
    model.eval()

    def one_batch() -> float:
        start = perf_counter()
        mu1_batch = sample_parameter_batch(mu1, mu1_perturb_std, param_batch_size)
        std1_batch = torch.clamp(
            sample_parameter_batch(std1, std1_perturb_std, param_batch_size), min=1e-4
        )
        mu2_batch = sample_parameter_batch(mu2, mu2_perturb_std, param_batch_size)
        std2_batch = torch.clamp(
            sample_parameter_batch(std2, std2_perturb_std, param_batch_size), min=1e-4
        )
        eps = torch.randn(param_batch_size, 3, device=device)
        samples = model(eps, mu1_batch, std1_batch, mu2_batch, std2_batch)
        norms = torch.linalg.norm(samples, dim=1)
        _ = norms.mean()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return perf_counter() - start

    for _ in range(warmup_batches):
        one_batch()

    rows: list[dict[str, float | int]] = []
    total_start = perf_counter()
    for batch_idx in range(k_batches):
        latency = one_batch()
        rows.append(
            {
                "batch_index": batch_idx,
                "param_batch_size": param_batch_size,
                "latency_seconds": latency,
                "latency_milliseconds": latency * 1e3,
                "samples_per_second": param_batch_size / latency,
            }
        )
    total_runtime = perf_counter() - total_start

    summary = {
        "k_batches": k_batches,
        "param_batch_size": param_batch_size,
        "total_latency_seconds": total_runtime,
        "total_latency_milliseconds": total_runtime * 1e3,
        "average_batch_latency_seconds": total_runtime / k_batches,
        "average_batch_latency_milliseconds": (total_runtime * 1e3) / k_batches,
        "total_samples": k_batches * param_batch_size,
        "effective_samples_per_second": (k_batches * param_batch_size) / total_runtime,
    }
    return rows, summary


def write_csv(rows: list[dict[str, float | int]], summary: dict[str, float | int], path: Path) -> None:
    """Write per-batch rows followed by a summary row."""
    with path.open("w", newline="", encoding="ascii") as handle:
        fieldnames = [
            "row_type",
            "batch_index",
            "param_batch_size",
            "latency_seconds",
            "latency_milliseconds",
            "samples_per_second",
            "k_batches",
            "total_latency_seconds",
            "total_latency_milliseconds",
            "average_batch_latency_seconds",
            "average_batch_latency_milliseconds",
            "total_samples",
            "effective_samples_per_second",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"row_type": "batch", **row})
        writer.writerow({"row_type": "summary", **summary})


def main() -> None:
    """Train once and benchmark K perturbed-parameter batches."""
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location="cpu", weights_only=True)
        if checkpoint["hidden1"] != args.hidden1 or checkpoint["hidden2"] != args.hidden2:
            raise RuntimeError(
                f"Checkpoint architecture ({checkpoint['hidden1']}, {checkpoint['hidden2']}) "
                f"does not match requested ({args.hidden1}, {args.hidden2})."
            )
        train_model = TwoLayerGaussianSampler(args.hidden1, args.hidden2)
        train_model.load_state_dict(checkpoint["state_dict"])
        train_model.eval()
        print(f"Loaded model from {args.load_model} (skipping training)")
    else:
        train_model = TwoLayerGaussianSampler(args.hidden1, args.hidden2).to(torch.device("cpu"))
        print("training NN surrogate on CPU...")
        train_surrogate(
            train_model,
            train_steps=args.train_steps,
            batch_size=args.train_batch_size,
            lr=args.lr,
            device=torch.device("cpu"),
        )
        if args.save_model:
            torch.save(
                {"hidden1": args.hidden1, "hidden2": args.hidden2, "state_dict": train_model.state_dict()},
                args.save_model,
            )
            print(f"Saved model to {args.save_model}")

    device = get_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended for GPU latency; use --device cuda.")

    model = clone_model_to_device(train_model, device)
    mu1 = parse_vec3(args.mu1).to(device)
    std1 = parse_vec3(args.std1).to(device)
    mu2 = parse_vec3(args.mu2).to(device)
    std2 = parse_vec3(args.std2).to(device)
    mu1_perturb_std = parse_vec3(args.mu1_perturb_std).to(device)
    std1_perturb_std = parse_vec3(args.std1_perturb_std).to(device)
    mu2_perturb_std = parse_vec3(args.mu2_perturb_std).to(device)
    std2_perturb_std = parse_vec3(args.std2_perturb_std).to(device)

    rows, summary = run_latency_benchmark(
        model,
        k_batches=args.k_batches,
        param_batch_size=args.param_batch_size,
        mu1=mu1,
        std1=std1,
        mu2=mu2,
        std2=std2,
        mu1_perturb_std=mu1_perturb_std,
        std1_perturb_std=std1_perturb_std,
        mu2_perturb_std=mu2_perturb_std,
        std2_perturb_std=std2_perturb_std,
        warmup_batches=args.warmup_batches,
    )

    print("\nPerturbed-parameter GPU latency benchmark")
    print(f"K batches:          {summary['k_batches']}")
    print(f"Parameter batch:    {summary['param_batch_size']:,}")
    print(f"Total latency:      {summary['total_latency_milliseconds']:.3f} ms")
    print(f"Average batch:      {summary['average_batch_latency_milliseconds']:.3f} ms")
    print(f"Effective samples/s {summary['effective_samples_per_second']:.3f}")

    output_path = Path(args.csv)
    write_csv(rows, summary, output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
