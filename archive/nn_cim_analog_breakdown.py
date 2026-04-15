"""Break down NN-surrogate inference on an analog-input CIM engine with analog RNG.

This is an analytical count model for the 3-stage Monte Carlo inference flow:
1. RNG  (analog hardware — finite throughput and energy, no digital cost)
2. NN surrogate executed on analog-input CIM arrays (no DAC; ADC readout only)
3. Euclidean distance

Assumptions:
- Analog RNG: Gaussian samples are generated in analog hardware on-chip.
  No digital operations or memory writes; characterised by throughput (Sa/s)
  and energy efficiency (J/Sa).
- Analog-input CIM array: inputs arrive in analog form; DAC conversion is
  eliminated. One ADC conversion is still counted per active output column
  per CIM tile use for readout.
- Tiling, weight memory, and feature/distance models are identical to
  nn_cim_inference_breakdown.py.

Throughput model (bottleneck of RNG and ADC):
    rng_normals_per_sample  = 3   (one 3-D Gaussian draw per MC sample)
    rng_limited  (samples/s) = rng_throughput / rng_normals_per_sample
    adc_limited  (samples/s) = adc_rate / adc_ops_per_sample
    throughput   (samples/s) = min(rng_limited, adc_limited)

Energy model (RNG contribution):
    rng_energy_per_sample (J) = rng_normals_per_sample * rng_efficiency
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


INPUT_DIM = 21
OUTPUT_DIM = 3
DEFAULT_CSV = "results_csv/nn_cim_analog_breakdown.csv"


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
    parser.add_argument(
        "--adc-rate",
        type=float,
        default=1e9,
        help="Aggregate ADC bandwidth in conversions/second (all ADC columns combined). Default: 1e9.",
    )
    parser.add_argument(
        "--rng-throughput",
        type=float,
        default=5.12e9,
        help="Analog RNG throughput in samples/second. Default: 5.12e9.",
    )
    parser.add_argument(
        "--rng-efficiency",
        type=float,
        default=0.36e-12,
        help="Analog RNG energy efficiency in J/sample. Default: 0.36e-12 (0.36 pJ/Sa).",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Output CSV path.")
    return parser


def _cim_layer_counts(
    in_dim: int,
    out_dim: int,
    num_samples: int,
    array_rows: int,
    array_cols: int,
) -> tuple[int, int, int, int]:
    """Return hardware-cost counts for one analog-input CIM layer.

    Tiling model:
    - row tiles = ceil(in_dim / array_rows)
    - col tiles = ceil(out_dim / array_cols)
    - each (row_tile, col_tile) pair uses one array instance for one sample
    - one ADC op is counted per active output column per tile use (no DAC)
    - row tiles beyond the first require digital accumulation of partial sums

    Returns:
        (cim_mac_ops, cim_array_uses, adc_ops, tile_accum_adds)
    """
    row_tiles = math.ceil(in_dim / array_rows)
    col_tiles = math.ceil(out_dim / array_cols)
    array_uses_per_sample = row_tiles * col_tiles

    adc_per_sample = 0
    for col_tile in range(col_tiles):
        remaining_cols = out_dim - (col_tile * array_cols)
        active_cols = min(array_cols, remaining_cols)
        adc_per_sample += row_tiles * active_cols

    cim_mac_per_sample = in_dim * out_dim
    tile_accum_adds_per_sample = max(0, row_tiles - 1) * out_dim
    return (
        num_samples * cim_mac_per_sample,
        num_samples * array_uses_per_sample,
        num_samples * adc_per_sample,
        num_samples * tile_accum_adds_per_sample,
    )


def rng_stage_counts(num_samples: int) -> StageCounts:
    """Return counts for analog Gaussian RNG (finite throughput/energy, no digital cost)."""
    return StageCounts(
        stage="RNG (analog)",
        rng_normals=3 * num_samples,
        multiplies=0,
        adds=0,
        sqrts=0,
        relus=0,
        cim_mac_ops=0,
        cim_array_uses=0,
        adc_ops=0,
        tile_accum_adds=0,
        reads=0,
        writes=0,
    )


def cim_surrogate_stage_counts(
    num_samples: int,
    hidden1: int,
    hidden2: int,
    array_rows: int,
    array_cols: int,
) -> StageCounts:
    """Return counts for feature construction plus analog-input CIM NN surrogate.

    Memory model:
    - NN weights are read once per Monte Carlo run.
    - Raw inputs are read once per sample.
    - Intermediate features and hidden activations stay on-chip.
    - Only the final 3D NN output is written for the next stage.
    """
    feature_multiplies = 6
    feature_adds = 6
    feature_sqrts = 3
    feature_reads = 15
    feature_writes = 21

    l1_mac, l1_uses, l1_adc, l1_tile_accum = _cim_layer_counts(
        INPUT_DIM, hidden1, num_samples, array_rows, array_cols
    )
    l2_mac, l2_uses, l2_adc, l2_tile_accum = _cim_layer_counts(
        hidden1, hidden2, num_samples, array_rows, array_cols
    )
    l3_mac, l3_uses, l3_adc, l3_tile_accum = _cim_layer_counts(
        hidden2, OUTPUT_DIM, num_samples, array_rows, array_cols
    )

    digital_adds_per_sample = hidden1 + hidden2 + OUTPUT_DIM
    relus_per_sample = hidden1 + hidden2

    per_run_weight_reads = (
        (INPUT_DIM * hidden1) + hidden1
        + (hidden1 * hidden2) + hidden2
        + (hidden2 * OUTPUT_DIM) + OUTPUT_DIM
    )
    per_sample_reads = feature_reads
    per_sample_writes = OUTPUT_DIM

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
        tile_accum_adds=0,
        reads=3 * num_samples,
        writes=1 * num_samples,
    )


def total_counts(rows: list[StageCounts]) -> StageCounts:
    """Aggregate stage counts."""
    return StageCounts(
        stage="Total",
        rng_normals=sum(r.rng_normals for r in rows),
        multiplies=sum(r.multiplies for r in rows),
        adds=sum(r.adds for r in rows),
        sqrts=sum(r.sqrts for r in rows),
        relus=sum(r.relus for r in rows),
        cim_mac_ops=sum(r.cim_mac_ops for r in rows),
        cim_array_uses=sum(r.cim_array_uses for r in rows),
        adc_ops=sum(r.adc_ops for r in rows),
        tile_accum_adds=sum(r.tile_accum_adds for r in rows),
        reads=sum(r.reads for r in rows),
        writes=sum(r.writes for r in rows),
    )


def print_table(
    rows: list[StageCounts],
    adc_rate: float,
    rng_throughput: float,
    rng_efficiency: float,
    num_samples: int,
) -> None:
    """Print a compact breakdown table followed by throughput and energy summary."""
    print("Analog-Input CIM NN Inference Breakdown")
    print(
        "stage               |  rng_normals |   multiplies |         adds |        sqrts |"
        "        relus |  cim_mac_ops | cim_array_uses |      adc_ops |"
        " tile_accum_adds |        reads |       writes"
    )
    print("-" * 200)
    for row in rows:
        print(
            f"{row.stage:<19} | {row.rng_normals:>12,} | {row.multiplies:>12,} | {row.adds:>12,} | "
            f"{row.sqrts:>12,} | {row.relus:>12,} | {row.cim_mac_ops:>12,} | {row.cim_array_uses:>14,} | "
            f"{row.adc_ops:>12,} | {row.tile_accum_adds:>15,} | "
            f"{row.reads:>12,} | {row.writes:>12,}"
        )

    total = rows[-1]
    rng_normals_per_sample = total.rng_normals / num_samples        # = 3
    adc_ops_per_sample = total.adc_ops / num_samples
    rng_limited = rng_throughput / rng_normals_per_sample
    adc_limited = adc_rate / adc_ops_per_sample
    throughput = min(rng_limited, adc_limited)
    bottleneck = "RNG" if rng_limited <= adc_limited else "ADC"
    rng_energy_per_sample = rng_normals_per_sample * rng_efficiency

    print()
    print(f"RNG throughput:         {rng_throughput:.3e} Sa/s  →  {rng_limited:.3e} MC samples/s")
    print(f"ADC rate (aggregate):   {adc_rate:.3e} conv/s  →  {adc_limited:.3e} MC samples/s")
    print(f"Bottleneck:             {bottleneck}")
    print(f"Throughput:             {throughput:.3e} samples/s")
    print(f"RNG energy/sample:      {rng_energy_per_sample * 1e12:.4f} pJ  ({rng_normals_per_sample:.0f} normals × {rng_efficiency*1e12:.2f} pJ/Sa)")


def write_csv(
    rows: list[StageCounts],
    adc_rate: float,
    rng_throughput: float,
    rng_efficiency: float,
    num_samples: int,
    output_path: Path,
) -> None:
    """Write the breakdown and throughput/energy summary to CSV."""
    fieldnames = [
        "stage",
        "rng_normals",
        "multiplies",
        "adds",
        "sqrts",
        "relus",
        "cim_mac_ops",
        "cim_array_uses",
        "adc_ops",
        "tile_accum_adds",
        "reads",
        "writes",
        "throughput_samples_per_sec",
        "bottleneck",
        "rng_energy_per_sample_pJ",
    ]
    total = rows[-1]
    rng_normals_per_sample = total.rng_normals / num_samples
    adc_ops_per_sample = total.adc_ops / num_samples
    rng_limited = rng_throughput / rng_normals_per_sample
    adc_limited = adc_rate / adc_ops_per_sample
    throughput = min(rng_limited, adc_limited)
    bottleneck = "RNG" if rng_limited <= adc_limited else "ADC"
    rng_energy_pj = rng_normals_per_sample * rng_efficiency * 1e12

    with output_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "stage": row.stage,
                "rng_normals": row.rng_normals,
                "multiplies": row.multiplies,
                "adds": row.adds,
                "sqrts": row.sqrts,
                "relus": row.relus,
                "cim_mac_ops": row.cim_mac_ops,
                "cim_array_uses": row.cim_array_uses,
                "adc_ops": row.adc_ops,
                "tile_accum_adds": row.tile_accum_adds,
                "reads": row.reads,
                "writes": row.writes,
                "throughput_samples_per_sec": "",
                "bottleneck": "",
                "rng_energy_per_sample_pJ": "",
            })
        writer.writerow({
            "stage": "Summary",
            "rng_normals": "",
            "multiplies": "",
            "adds": "",
            "sqrts": "",
            "relus": "",
            "cim_mac_ops": "",
            "cim_array_uses": "",
            "adc_ops": "",
            "tile_accum_adds": "",
            "reads": "",
            "writes": "",
            "throughput_samples_per_sec": throughput,
            "bottleneck": bottleneck,
            "rng_energy_per_sample_pJ": rng_energy_pj,
        })


def main() -> None:
    """Generate the analog-input CIM breakdown and throughput estimate."""
    args = build_parser().parse_args()
    if min(args.hidden1, args.hidden2, args.num_samples, args.array_rows, args.array_cols) <= 0:
        raise ValueError("All sizes must be positive.")
    if args.adc_rate <= 0:
        raise ValueError("--adc-rate must be positive.")
    if args.rng_throughput <= 0:
        raise ValueError("--rng-throughput must be positive.")
    if args.rng_efficiency <= 0:
        raise ValueError("--rng-efficiency must be positive.")

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

    print_table(rows, args.adc_rate, args.rng_throughput, args.rng_efficiency, args.num_samples)
    output_path = Path(args.csv)
    write_csv(rows, args.adc_rate, args.rng_throughput, args.rng_efficiency, args.num_samples, output_path)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
