"""Break down GPU NN inference into RNG, NN surrogate, and Euclidean distance.

This script reports estimated logical operation counts for the GPU inference
path in ``nn_gaussian_distance.py``.

Stages:
1. RNG: generate eps ~ N(0, I_3)
2. NN surrogate: build features and run the two-layer ReLU network
3. Euclidean distance: compute ||sample||_2

Outputs:
- console table
- CSV file
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CSV = "results_csv/nn_gpu_inference_breakdown.csv"


@dataclass(frozen=True)
class StageCounts:
    """Logical operation counts for one stage."""

    stage: str
    rng_normals: int
    multiplies: int
    adds: int
    sqrts: int
    relus: int
    reads: int
    writes: int


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden1", type=int, default=64, help="First hidden layer width.")
    parser.add_argument("--hidden2", type=int, default=64, help="Second hidden layer width.")
    parser.add_argument("--num-samples", type=int, default=1_000_000, help="Total Monte Carlo samples.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Output CSV path.")
    return parser


def rng_stage_counts(num_samples: int) -> StageCounts:
    """Return counts for the Gaussian RNG stage."""
    return StageCounts(
        stage="RNG",
        rng_normals=3 * num_samples,
        multiplies=0,
        adds=0,
        sqrts=0,
        relus=0,
        reads=0,
        writes=3 * num_samples,
    )


def nn_surrogate_stage_counts(num_samples: int, hidden1: int, hidden2: int) -> StageCounts:
    """Return counts for feature construction plus NN forward inference.

    Assumption:
    - NN parameters are loaded once per Monte Carlo run, not once per sample.
    - Feature tensors and activations are still counted per sample.
    """
    feature_multiplies = 6
    feature_adds = 6
    feature_sqrts = 3
    feature_reads = 15
    feature_writes = 21

    layer1_multiplies = 21 * hidden1
    layer1_adds = 21 * hidden1
    layer1_relus = hidden1
    layer1_weight_reads = (21 * hidden1) + hidden1
    layer1_activation_reads = 21
    layer1_writes = hidden1

    layer2_multiplies = hidden1 * hidden2
    layer2_adds = hidden1 * hidden2
    layer2_relus = hidden2
    layer2_weight_reads = (hidden1 * hidden2) + hidden2
    layer2_activation_reads = hidden1
    layer2_writes = hidden2

    output_multiplies = 3 * hidden2
    output_adds = 3 * hidden2
    output_weight_reads = (3 * hidden2) + 3
    output_activation_reads = hidden2
    output_writes = 3

    multiplies = (
        feature_multiplies
        + layer1_multiplies
        + layer2_multiplies
        + output_multiplies
    )
    adds = feature_adds + layer1_adds + layer2_adds + output_adds
    sqrts = feature_sqrts
    relus = layer1_relus + layer2_relus
    per_run_weight_reads = layer1_weight_reads + layer2_weight_reads + output_weight_reads
    per_sample_reads = (
        feature_reads
        + layer1_activation_reads
        + layer2_activation_reads
        + output_activation_reads
    )
    reads = per_run_weight_reads + (num_samples * per_sample_reads)
    writes = num_samples * (feature_writes + layer1_writes + layer2_writes + output_writes)

    return StageCounts(
        stage="NN surrogate",
        rng_normals=0,
        multiplies=num_samples * multiplies,
        adds=num_samples * adds,
        sqrts=num_samples * sqrts,
        relus=num_samples * relus,
        reads=reads,
        writes=writes,
    )


def distance_stage_counts(num_samples: int) -> StageCounts:
    """Return counts for Euclidean norm computation."""
    return StageCounts(
        stage="Euclidean distance",
        rng_normals=0,
        multiplies=3 * num_samples,
        adds=2 * num_samples,
        sqrts=1 * num_samples,
        relus=0,
        reads=3 * num_samples,
        writes=1 * num_samples,
    )


def total_counts(stages: list[StageCounts]) -> StageCounts:
    """Aggregate counts across stages."""
    return StageCounts(
        stage="Total",
        rng_normals=sum(stage.rng_normals for stage in stages),
        multiplies=sum(stage.multiplies for stage in stages),
        adds=sum(stage.adds for stage in stages),
        sqrts=sum(stage.sqrts for stage in stages),
        relus=sum(stage.relus for stage in stages),
        reads=sum(stage.reads for stage in stages),
        writes=sum(stage.writes for stage in stages),
    )


def print_table(rows: list[StageCounts]) -> None:
    """Print a compact table."""
    print("GPU NN Inference Breakdown")
    print("stage               |  rng_normals |   multiplies |         adds |        sqrts |        relus |        reads |       writes")
    print("-" * 140)
    for row in rows:
        print(
            f"{row.stage:<19} | {row.rng_normals:>12,} | {row.multiplies:>12,} | {row.adds:>12,} | "
            f"{row.sqrts:>12,} | {row.relus:>12,} | {row.reads:>12,} | {row.writes:>12,}"
        )


def write_csv(rows: list[StageCounts], output_path: Path) -> None:
    """Write rows to CSV."""
    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "rng_normals",
                "multiplies",
                "adds",
                "sqrts",
                "relus",
                "reads",
                "writes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row.stage,
                    "rng_normals": row.rng_normals,
                    "multiplies": row.multiplies,
                    "adds": row.adds,
                    "sqrts": row.sqrts,
                    "relus": row.relus,
                    "reads": row.reads,
                    "writes": row.writes,
                }
            )


def main() -> None:
    """Generate the breakdown table and CSV."""
    args = build_parser().parse_args()
    if args.hidden1 <= 0 or args.hidden2 <= 0 or args.num_samples <= 0:
        raise ValueError("hidden layer sizes and num_samples must be positive.")

    rows = [
        rng_stage_counts(args.num_samples),
        nn_surrogate_stage_counts(args.num_samples, args.hidden1, args.hidden2),
        distance_stage_counts(args.num_samples),
    ]
    rows.append(total_counts(rows))

    print_table(rows)
    output_path = Path(args.csv)
    write_csv(rows, output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
