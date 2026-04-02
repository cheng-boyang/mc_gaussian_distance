"""Break down NN-surrogate inference using a compute-in-memory (CIM) engine.

This is an analytical count model for the 3-stage Monte Carlo inference flow:
1. RNG
2. NN surrogate executed on CIM arrays
3. Euclidean distance

The NN matches ``nn_gaussian_distance.py``:
- input dimension: 21
- hidden layers: configurable
- output dimension: 3
- activation: ReLU

Assumptions for the CIM surrogate:
- A matrix-vector multiply is tiled onto CIM arrays of size
  ``array_rows x array_cols``.
- Each CIM array can perform MACs for one tile simultaneously.
- One ADC conversion is counted per active output column per CIM tile use.
- NN weights are loaded once per Monte Carlo run, not once per sample.
- Feature construction and Euclidean norm are still conventional digital logic.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


INPUT_DIM = 21
OUTPUT_DIM = 3
DEFAULT_CSV = "results_csv/nn_cim_inference_breakdown.csv"


@dataclass(frozen=True)
class StageCounts:
    """Logical operation counts for one stage."""

    stage: str
    rng_normals: int
    multiplies: int
    adds: int
    sqrts: int
    relus: int
    cim_mac_ops: int
    cim_array_uses: int
    adc_ops: int
    dac_ops: int
    tile_accum_adds: int
    reads: int
    writes: int


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden1", type=int, default=64, help="First hidden layer width.")
    parser.add_argument("--hidden2", type=int, default=64, help="Second hidden layer width.")
    parser.add_argument("--num-samples", type=int, default=1_000_000, help="Total Monte Carlo samples.")
    parser.add_argument("--array-rows", type=int, default=128, help="CIM array row count.")
    parser.add_argument("--array-cols", type=int, default=128, help="CIM array column count.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Output CSV path.")
    return parser


def _cim_layer_counts(
    in_dim: int,
    out_dim: int,
    num_samples: int,
    array_rows: int,
    array_cols: int,
) -> tuple[int, int, int, int, int]:
    """Return hardware-cost counts for one CIM layer.

    Tiling model:
    - row tiles = ceil(in_dim / array_rows)
    - col tiles = ceil(out_dim / array_cols)
    - each (row_tile, col_tile) pair uses one array instance for one sample
    - each tile performs up to array_rows * active_cols MACs simultaneously
    - one ADC op is counted per active output column per tile use
    - one DAC op is counted per active input row per tile use
    - row tiles beyond the first require digital accumulation of partial sums
    """
    row_tiles = math.ceil(in_dim / array_rows)
    col_tiles = math.ceil(out_dim / array_cols)
    array_uses_per_sample = row_tiles * col_tiles

    adc_per_sample = 0
    dac_per_sample = 0
    for col_tile in range(col_tiles):
        remaining_cols = out_dim - (col_tile * array_cols)
        active_cols = min(array_cols, remaining_cols)
        for row_tile in range(row_tiles):
            remaining_rows = in_dim - (row_tile * array_rows)
            active_rows = min(array_rows, remaining_rows)
            dac_per_sample += active_rows
        adc_per_sample += row_tiles * active_cols

    cim_mac_per_sample = in_dim * out_dim
    tile_accum_adds_per_sample = max(0, row_tiles - 1) * out_dim
    return (
        num_samples * cim_mac_per_sample,
        num_samples * array_uses_per_sample,
        num_samples * adc_per_sample,
        num_samples * dac_per_sample,
        num_samples * tile_accum_adds_per_sample,
    )


def rng_stage_counts(num_samples: int) -> StageCounts:
    """Return counts for Gaussian RNG."""
    return StageCounts(
        stage="RNG",
        rng_normals=3 * num_samples,
        multiplies=0,
        adds=0,
        sqrts=0,
        relus=0,
        cim_mac_ops=0,
        cim_array_uses=0,
        adc_ops=0,
        dac_ops=0,
        tile_accum_adds=0,
        reads=0,
        writes=3 * num_samples,
    )


def cim_surrogate_stage_counts(
    num_samples: int,
    hidden1: int,
    hidden2: int,
    array_rows: int,
    array_cols: int,
) -> StageCounts:
    """Return counts for feature construction plus CIM-executed NN surrogate."""
    feature_multiplies = 6
    feature_adds = 6
    feature_sqrts = 3
    feature_reads = 15
    feature_writes = 21

    l1_mac, l1_uses, l1_adc, l1_dac, l1_tile_accum = _cim_layer_counts(
        INPUT_DIM, hidden1, num_samples, array_rows, array_cols
    )
    l2_mac, l2_uses, l2_adc, l2_dac, l2_tile_accum = _cim_layer_counts(
        hidden1, hidden2, num_samples, array_rows, array_cols
    )
    l3_mac, l3_uses, l3_adc, l3_dac, l3_tile_accum = _cim_layer_counts(
        hidden2, OUTPUT_DIM, num_samples, array_rows, array_cols
    )

    digital_adds_per_sample = hidden1 + hidden2 + OUTPUT_DIM
    relus_per_sample = hidden1 + hidden2

    per_run_weight_reads = (
        (INPUT_DIM * hidden1) + hidden1
        + (hidden1 * hidden2) + hidden2
        + (hidden2 * OUTPUT_DIM) + OUTPUT_DIM
    )
    per_sample_reads = feature_reads + INPUT_DIM + hidden1 + hidden2
    per_sample_writes = feature_writes + hidden1 + hidden2 + OUTPUT_DIM

    return StageCounts(
        stage="NN surrogate (CIM)",
        rng_normals=0,
        multiplies=num_samples * feature_multiplies,
        adds=num_samples * (feature_adds + digital_adds_per_sample)
        + l1_tile_accum
        + l2_tile_accum
        + l3_tile_accum,
        sqrts=num_samples * feature_sqrts,
        relus=num_samples * relus_per_sample,
        cim_mac_ops=l1_mac + l2_mac + l3_mac,
        cim_array_uses=l1_uses + l2_uses + l3_uses,
        adc_ops=l1_adc + l2_adc + l3_adc,
        dac_ops=l1_dac + l2_dac + l3_dac,
        tile_accum_adds=l1_tile_accum + l2_tile_accum + l3_tile_accum,
        reads=per_run_weight_reads + (num_samples * per_sample_reads),
        writes=num_samples * per_sample_writes,
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
        cim_mac_ops=0,
        cim_array_uses=0,
        adc_ops=0,
        dac_ops=0,
        tile_accum_adds=0,
        reads=3 * num_samples,
        writes=1 * num_samples,
    )


def total_counts(rows: list[StageCounts]) -> StageCounts:
    """Aggregate stage counts."""
    return StageCounts(
        stage="Total",
        rng_normals=sum(row.rng_normals for row in rows),
        multiplies=sum(row.multiplies for row in rows),
        adds=sum(row.adds for row in rows),
        sqrts=sum(row.sqrts for row in rows),
        relus=sum(row.relus for row in rows),
        cim_mac_ops=sum(row.cim_mac_ops for row in rows),
        cim_array_uses=sum(row.cim_array_uses for row in rows),
        adc_ops=sum(row.adc_ops for row in rows),
        dac_ops=sum(row.dac_ops for row in rows),
        tile_accum_adds=sum(row.tile_accum_adds for row in rows),
        reads=sum(row.reads for row in rows),
        writes=sum(row.writes for row in rows),
    )


def print_table(rows: list[StageCounts]) -> None:
    """Print a compact table."""
    print("CIM NN Inference Breakdown")
    print(
        "stage               |  rng_normals |   multiplies |         adds |        sqrts |"
        "        relus |  cim_mac_ops | cim_array_uses |      adc_ops |"
        "      dac_ops | tile_accum_adds |        reads |       writes"
    )
    print("-" * 214)
    for row in rows:
        print(
            f"{row.stage:<19} | {row.rng_normals:>12,} | {row.multiplies:>12,} | {row.adds:>12,} | "
            f"{row.sqrts:>12,} | {row.relus:>12,} | {row.cim_mac_ops:>12,} | {row.cim_array_uses:>14,} | "
            f"{row.adc_ops:>12,} | {row.dac_ops:>12,} | {row.tile_accum_adds:>15,} | "
            f"{row.reads:>12,} | {row.writes:>12,}"
        )


def write_csv(rows: list[StageCounts], output_path: Path) -> None:
    """Write the breakdown to CSV."""
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
                "cim_mac_ops",
                "cim_array_uses",
                "adc_ops",
                "dac_ops",
                "tile_accum_adds",
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
                    "cim_mac_ops": row.cim_mac_ops,
                    "cim_array_uses": row.cim_array_uses,
                    "adc_ops": row.adc_ops,
                    "dac_ops": row.dac_ops,
                    "tile_accum_adds": row.tile_accum_adds,
                    "reads": row.reads,
                    "writes": row.writes,
                }
            )


def main() -> None:
    """Generate the CIM breakdown."""
    args = build_parser().parse_args()
    if min(args.hidden1, args.hidden2, args.num_samples, args.array_rows, args.array_cols) <= 0:
        raise ValueError("All sizes must be positive.")

    rows = [
        rng_stage_counts(args.num_samples),
        cim_surrogate_stage_counts(
            args.num_samples,
            args.hidden1,
            args.hidden2,
            args.array_rows,
            args.array_cols,
        ),
        distance_stage_counts(args.num_samples),
    ]
    rows.append(total_counts(rows))

    print_table(rows)
    output_path = Path(args.csv)
    write_csv(rows, output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
