"""flies_in_the_room.py — Voxel-based Gaussian distance estimation.

Scenario
--------
A 3D grid of grid_size^3 voxels, each storing:
    [mu_x, mu_y, mu_z, std_x, std_y, std_z]   (diagonal Gaussian)

Each "run" randomly selects two voxels, fetches their Gaussian parameters,
then estimates E[||Z||_2] where Z = X - Y ~ N(mu1-mu2, diag(std1^2+std2^2))
using the trained NN surrogate via Monte Carlo averaging.

NN surrogate
------------
Input  (15D): eps(3) + mu1(3) + std1(3) + mu2(3) + std2(3)
Output ( 3D): sample from Z = X - Y (no final activation)
MC over N eps draws → mean of ||Z||_2 norms = distance estimate for the run.
Training target: exact Z sample = (mu1-mu2) + eps*sqrt(std1^2+std2^2)

Parameter range advice
----------------------
mu_range  : half-width for mu ~ Uniform[-mu_range, mu_range].
            Default 5.0. For a 100^3 grid at unit spacing, this puts most
            inter-voxel distances in a physically interesting regime.
std_min / std_max : diagonal std ~ Uniform[std_min, std_max].
            Default [0.1, 2.0]. Keeps distributions non-degenerate and
            overlapping enough that the distance varies meaningfully.

Key metrics
-----------
- Accuracy : mean absolute error and mean relative error of NN vs exact MC
- Throughput: CPU and GPU (measured, in MC samples/s), CIM (analytical)
- Energy    : analog RNG contribution per MC sample (pJ)
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from gaussian_distance_cpu import deterministic_expected_norm_cpu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_DIM = 15    # eps(3) + mu1(3) + std1(3) + mu2(3) + std2(3)
VOXEL_PARAMS = 6  # mu(3) + std(3)


# ---------------------------------------------------------------------------
# Voxel map
# ---------------------------------------------------------------------------

def generate_voxel_map(
    grid_size: int,
    mu_range: float,
    std_min: float,
    std_max: float,
    *,
    seed: int,
) -> torch.Tensor:
    """Return a (grid_size, grid_size, grid_size, 6) CPU float32 tensor.

    Channels [:3] = mu ~ Uniform[-mu_range, mu_range]
    Channels [3:] = std ~ Uniform[std_min, std_max]
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    vmap = torch.empty(grid_size, grid_size, grid_size, VOXEL_PARAMS, dtype=torch.float32)
    vmap[..., :3].uniform_(-mu_range, mu_range, generator=gen)
    vmap[..., 3:].uniform_(std_min, std_max, generator=gen)
    return vmap


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class TwoLayerDistanceNet(nn.Module):
    """Two-layer ReLU net: (eps, mu1, std1, mu2, std2) -> 3D sample Z from Z = X - Y."""

    def __init__(self, hidden1: int, hidden2: int) -> None:
        super().__init__()
        if hidden1 <= 0 or hidden2 <= 0:
            raise ValueError("hidden layer sizes must be positive.")
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 3),
        )

    def forward(
        self,
        eps: torch.Tensor,    # (N, 3)
        mu1: torch.Tensor,    # (N, 3)
        std1: torch.Tensor,   # (N, 3)
        mu2: torch.Tensor,    # (N, 3)
        std2: torch.Tensor,   # (N, 3)
    ) -> torch.Tensor:        # (N, 3)  — samples from Z = X - Y
        features = torch.cat([eps, mu1, std1, mu2, std2], dim=-1)
        return self.net(features)


def exact_z_samples(
    eps: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
) -> torch.Tensor:
    """Exact samples from Z = X - Y ~ N(mu1-mu2, diag(std1^2+std2^2))."""
    return (mu1 - mu2) + eps * torch.sqrt(std1.square() + std2.square())


# ---------------------------------------------------------------------------
# Training  (hand-rolled Adam, consistent with rest of codebase)
# ---------------------------------------------------------------------------

def _sample_train_batch(
    batch_size: int,
    mu_range: float,
    std_min: float,
    std_max: float,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    mu1 = torch.empty(batch_size, 3, device=device).uniform_(-mu_range, mu_range)
    mu2 = torch.empty(batch_size, 3, device=device).uniform_(-mu_range, mu_range)
    std1 = torch.empty(batch_size, 3, device=device).uniform_(std_min, std_max)
    std2 = torch.empty(batch_size, 3, device=device).uniform_(std_min, std_max)
    eps = torch.randn(batch_size, 3, device=device)
    return eps, mu1, std1, mu2, std2


def train_net(
    model: TwoLayerDistanceNet,
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    mu_range: float,
    std_min: float,
    std_max: float,
    device: torch.device,
) -> None:
    """Train to predict exact ||Z||_2 per eps draw."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    m_buf = [torch.zeros_like(p) for p in params]
    v_buf = [torch.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for step in range(train_steps):
        eps, mu1, std1, mu2, std2 = _sample_train_batch(batch_size, mu_range, std_min, std_max, device)
        target = exact_z_samples(eps, mu1, std1, mu2, std2)
        pred = model(eps, mu1, std1, mu2, std2)
        loss = torch.mean((pred - target) ** 2)

        for p in model.parameters():
            p.grad = None
        loss.backward()

        with torch.no_grad():
            idx = step + 1
            bc1 = 1.0 - beta1 ** idx
            bc2 = 1.0 - beta2 ** idx
            for p, mi, vi in zip(params, m_buf, v_buf):
                if p.grad is None:
                    continue
                g = p.grad
                mi.mul_(beta1).add_(g, alpha=1.0 - beta1)
                vi.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                p.addcdiv_(mi / bc1, torch.sqrt(vi / bc2).add_(eps_adam), value=-lr)

        if (step + 1) % max(1, train_steps // 10) == 0:
            print(f"  step {step + 1:>6d}/{train_steps}: mse={float(loss.item()):.6e}")


# ---------------------------------------------------------------------------
# K-run benchmark
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_idx: int
    device: str
    nn_estimate: float
    exact_estimate: float
    abs_error: float
    rel_error: float
    nn_time_s: float
    exact_time_s: float


def _clone_to(model: TwoLayerDistanceNet, device: torch.device) -> TwoLayerDistanceNet:
    h1 = model.net[0].out_features
    h2 = model.net[2].out_features
    clone = TwoLayerDistanceNet(h1, h2).to(device)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    return clone


@torch.no_grad()
def run_benchmark(
    model: TwoLayerDistanceNet,
    voxel_map: torch.Tensor,
    *,
    k_runs: int,
    n_samples: int,
    grid_size: int,
    device: torch.device,
    device_label: str,
    quadrature_order: int = 15,
    warmup_runs: int = 2,
) -> list[RunResult]:
    """Run K random voxel-pair evaluations, each with N MC samples.

    Ground truth is Gauss-Hermite quadrature (deterministic), not MC.
    """
    vmap_dev = voxel_map.to(device)

    def _one_run(measure: bool) -> RunResult | None:
        coords = torch.randint(0, grid_size, (2, 3), device=device)
        p1 = vmap_dev[coords[0, 0], coords[0, 1], coords[0, 2]]
        p2 = vmap_dev[coords[1, 0], coords[1, 1], coords[1, 2]]
        mu1 = p1[:3].unsqueeze(0).expand(n_samples, -1)
        std1 = p1[3:].unsqueeze(0).expand(n_samples, -1)
        mu2 = p2[:3].unsqueeze(0).expand(n_samples, -1)
        std2 = p2[3:].unsqueeze(0).expand(n_samples, -1)
        eps = torch.randn(n_samples, 3, device=device)

        t0 = perf_counter()
        nn_samples = model(eps, mu1, std1, mu2, std2)          # (N, 3)
        nn_norms = torch.linalg.norm(nn_samples, dim=-1)       # (N,)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        nn_time = perf_counter() - t0
        nn_est = float(nn_norms.mean())

        # Deterministic ground truth via Gauss-Hermite quadrature (CPU, numpy)
        mu1_np = p1[:3].cpu().numpy().astype(np.float64)
        std1_np = p1[3:].cpu().numpy().astype(np.float64)
        mu2_np = p2[:3].cpu().numpy().astype(np.float64)
        std2_np = p2[3:].cpu().numpy().astype(np.float64)
        mu_z = mu1_np - mu2_np
        sigma_z = np.diag(std1_np ** 2 + std2_np ** 2)
        t0 = perf_counter()
        det_result = deterministic_expected_norm_cpu(mu_z, sigma_z, quadrature_order=quadrature_order)
        det_time = perf_counter() - t0
        det_est = det_result.estimate

        if not measure:
            return None
        abs_err = abs(nn_est - det_est)
        rel_err = abs_err / max(det_est, 1e-12)
        return RunResult(
            run_idx=0,
            device=device_label,
            nn_estimate=nn_est,
            exact_estimate=det_est,
            abs_error=abs_err,
            rel_error=rel_err,
            nn_time_s=nn_time,
            exact_time_s=det_time,
        )

    for _ in range(warmup_runs):
        _one_run(measure=False)

    results = []
    for k in range(k_runs):
        r = _one_run(measure=True)
        r.run_idx = k
        results.append(r)
    return results


def _summarize(results: list[RunResult], n_samples: int) -> dict:
    k = len(results)
    mae = sum(r.abs_error for r in results) / k
    mre = sum(r.rel_error for r in results) / k
    total_nn_time = sum(r.nn_time_s for r in results)
    tput = n_samples * k / total_nn_time
    return {
        "device": results[0].device,
        "k_runs": k,
        "n_samples": n_samples,
        "mean_abs_error": mae,
        "mean_rel_error_pct": mre * 100.0,
        "nn_throughput_samples_per_sec": tput,
        "nn_throughput_Msamples_per_sec": tput / 1e6,
    }


# ---------------------------------------------------------------------------
# CIM analytical model  (analog RNG + no-DAC array)
# ---------------------------------------------------------------------------

def _cim_layer_adc(in_dim: int, out_dim: int, array_rows: int, array_cols: int) -> int:
    """ADC operations per MC sample for one linear layer."""
    row_tiles = math.ceil(in_dim / array_rows)
    col_tiles = math.ceil(out_dim / array_cols)
    adc = 0
    for ct in range(col_tiles):
        active_cols = min(array_cols, out_dim - ct * array_cols)
        adc += row_tiles * active_cols
    return adc


def cim_throughput_estimate(
    hidden1: int,
    hidden2: int,
    *,
    array_rows: int,
    array_cols: int,
    adc_rate: float,
    rng_throughput: float,
    rng_efficiency: float,
) -> dict:
    """Analytical CIM throughput for the flies_in_the_room NN.

    NN layers: Linear(15->h1), Linear(h1->h2), Linear(h2->1)
    Analog inputs (no DAC), ADC readout, analog RNG for eps.
    """
    rng_per_sample = 3   # eps ~ N(0, I_3)

    l1_out = 1  # output dim of layer 3
    adc_l1 = _cim_layer_adc(INPUT_DIM, hidden1, array_rows, array_cols)
    adc_l2 = _cim_layer_adc(hidden1, hidden2, array_rows, array_cols)
    adc_l3 = _cim_layer_adc(hidden2, l1_out, array_rows, array_cols)
    total_adc = adc_l1 + adc_l2 + adc_l3

    cim_mac = INPUT_DIM * hidden1 + hidden1 * hidden2 + hidden2 * l1_out

    rng_limited = rng_throughput / rng_per_sample
    adc_limited = adc_rate / total_adc
    throughput = min(rng_limited, adc_limited)
    bottleneck = "RNG" if rng_limited <= adc_limited else "ADC"
    rng_energy_pj = rng_per_sample * rng_efficiency * 1e12

    return {
        "array_rows": array_rows,
        "array_cols": array_cols,
        "cim_mac_per_sample": cim_mac,
        "adc_ops_per_sample": total_adc,
        "rng_normals_per_sample": rng_per_sample,
        "rng_limited_Msamples_per_sec": rng_limited / 1e6,
        "adc_limited_Msamples_per_sec": adc_limited / 1e6,
        "cim_throughput_Msamples_per_sec": throughput / 1e6,
        "bottleneck": bottleneck,
        "rng_energy_per_sample_pJ": rng_energy_pj,
    }


# ---------------------------------------------------------------------------
# Histogram plot
# ---------------------------------------------------------------------------

def plot_histogram(
    results: list[RunResult],
    output_path: Path,
    bins: int = 30,
) -> None:
    """Histogram of exact MC distances vs NN estimates across K runs."""
    exact_vals = [r.exact_estimate for r in results]
    nn_vals = [r.nn_estimate for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: overlapping histograms
    ax = axes[0]
    ax.hist(exact_vals, bins=bins, alpha=0.6, label="Deterministic (quadrature)", color="steelblue", edgecolor="white")
    ax.hist(nn_vals,    bins=bins, alpha=0.6, label="NN estimate", color="tomato",    edgecolor="white")
    ax.set_xlabel("Distance estimate")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of distance estimates over K runs")
    ax.legend()

    # Right: scatter exact vs NN with diagonal
    ax2 = axes[1]
    ax2.scatter(exact_vals, nn_vals, s=12, alpha=0.6, color="steelblue")
    lo = min(min(exact_vals), min(nn_vals))
    hi = max(max(exact_vals), max(nn_vals))
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax2.set_xlabel("Deterministic distance (quadrature)")
    ax2.set_ylabel("NN estimate")
    ax2.set_title("NN vs Deterministic ground truth (per run)")
    ax2.legend()

    device_label = results[0].device
    mae = sum(r.abs_error for r in results) / len(results)
    mre = sum(r.rel_error for r in results) / len(results) * 100
    fig.suptitle(
        f"{device_label}  |  K={len(results)} runs  |  MAE={mae:.4f}  |  Rel err={mre:.2f}%",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved histogram → {output_path}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(
    run_results: list[RunResult],
    summaries: list[dict],
    cim_stats: dict,
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="ascii") as fh:
        # Section 1: per-run rows
        run_fields = ["row_type", "device", "run_idx", "nn_estimate", "exact_estimate",
                      "abs_error", "rel_error", "nn_time_s", "exact_time_s"]
        w = csv.DictWriter(fh, fieldnames=run_fields)
        w.writeheader()
        for r in run_results:
            w.writerow({"row_type": "run", "device": r.device, "run_idx": r.run_idx,
                        "nn_estimate": r.nn_estimate, "exact_estimate": r.exact_estimate,
                        "abs_error": r.abs_error, "rel_error": r.rel_error,
                        "nn_time_s": r.nn_time_s, "exact_time_s": r.exact_time_s})

        # Section 2: device summaries
        fh.write("\n")
        sum_fields = ["row_type", "device", "k_runs", "n_samples",
                      "mean_abs_error", "mean_rel_error_pct",
                      "nn_throughput_samples_per_sec", "nn_throughput_Msamples_per_sec"]
        w2 = csv.DictWriter(fh, fieldnames=sum_fields)
        w2.writeheader()
        for s in summaries:
            w2.writerow({"row_type": "summary", **s})

        # Section 3: CIM analytical
        fh.write("\n")
        cim_fields = ["row_type"] + list(cim_stats.keys())
        w3 = csv.DictWriter(fh, fieldnames=cim_fields)
        w3.writeheader()
        w3.writerow({"row_type": "cim", **cim_stats})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # Voxel map
    p.add_argument("--grid-size", type=int, default=100,
                   help="Voxels per dimension (default: 100).")
    p.add_argument("--mu-range", type=float, default=5.0,
                   help="mu ~ Uniform[-mu_range, mu_range] (default: 5.0).")
    p.add_argument("--std-min", type=float, default=0.1,
                   help="Diagonal std lower bound (default: 0.1).")
    p.add_argument("--std-max", type=float, default=2.0,
                   help="Diagonal std upper bound (default: 2.0).")
    # NN architecture
    p.add_argument("--hidden1", type=int, default=64, help="First hidden layer width.")
    p.add_argument("--hidden2", type=int, default=64, help="Second hidden layer width.")
    # Training
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--train-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    # Benchmark
    p.add_argument("--n-samples", type=int, default=1000,
                   help="MC samples per run (N, default: 1000).")
    p.add_argument("--k-runs", type=int, default=100,
                   help="Number of runs (K, default: 100).")
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--quadrature-order", type=int, default=15,
                   help="Gauss-Hermite quadrature order for ground truth (default: 15, gives 15^3=3375 points).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto", help="cuda / cpu / auto.")
    # CIM hardware parameters
    p.add_argument("--array-rows", type=int, default=64)
    p.add_argument("--array-cols", type=int, default=64)
    p.add_argument("--adc-rate", type=float, default=1e9,
                   help="Aggregate ADC bandwidth in conv/s (default: 1e9).")
    p.add_argument("--rng-throughput", type=float, default=5.12e9,
                   help="Analog RNG throughput in Sa/s (default: 5.12e9).")
    p.add_argument("--rng-efficiency", type=float, default=0.36e-12,
                   help="Analog RNG energy in J/Sa (default: 0.36e-12).")
    # Output
    p.add_argument("--csv", default="results_csv/flies_in_the_room.csv")
    p.add_argument("--plot", default="results_csv/flies_in_the_room.png",
                   help="Path for the histogram PNG (default: results_csv/flies_in_the_room.png).")
    p.add_argument("--hist-bins", type=int, default=30, help="Number of histogram bins (default: 30).")
    return p


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    train_device = get_device(args.device)

    # 1. Voxel map
    print(f"Generating {args.grid_size}^3 voxel map...")
    voxel_map = generate_voxel_map(
        args.grid_size, args.mu_range, args.std_min, args.std_max, seed=args.seed,
    )
    mb = voxel_map.numel() * 4 / 1e6
    print(f"  shape: {tuple(voxel_map.shape)}  ({mb:.1f} MB)")
    print(f"  mu ~ Uniform[-{args.mu_range}, {args.mu_range}]")
    print(f"  std ~ Uniform[{args.std_min}, {args.std_max}]")

    # 2. Train
    print(f"\nTraining NN ({INPUT_DIM}→{args.hidden1}→{args.hidden2}→3) on {train_device}...")
    model = TwoLayerDistanceNet(args.hidden1, args.hidden2).to(train_device)
    train_net(
        model,
        train_steps=args.train_steps,
        batch_size=args.train_batch_size,
        lr=args.lr,
        mu_range=args.mu_range,
        std_min=args.std_min,
        std_max=args.std_max,
        device=train_device,
    )

    # 3. Benchmark GPU
    all_run_results: list[RunResult] = []
    summaries: list[dict] = []
    gpu_results: list[RunResult] = []

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_model = _clone_to(model, gpu_device)
        print(f"\nGPU benchmark: K={args.k_runs} runs, N={args.n_samples:,} samples each...")
        gpu_results = run_benchmark(
            gpu_model, voxel_map,
            k_runs=args.k_runs, n_samples=args.n_samples,
            grid_size=args.grid_size, device=gpu_device,
            device_label="GPU", quadrature_order=args.quadrature_order,
            warmup_runs=args.warmup_runs,
        )
        all_run_results.extend(gpu_results)
        s = _summarize(gpu_results, args.n_samples)
        summaries.append(s)
        print(f"  MAE:        {s['mean_abs_error']:.6f}")
        print(f"  Rel error:  {s['mean_rel_error_pct']:.3f}%")
        print(f"  Throughput: {s['nn_throughput_Msamples_per_sec']:.3f} M samples/s")

    # 4. Benchmark CPU
    cpu_device = torch.device("cpu")
    cpu_model = _clone_to(model, cpu_device)
    print(f"\nCPU benchmark: K={args.k_runs} runs, N={args.n_samples:,} samples each...")
    cpu_results = run_benchmark(
        cpu_model, voxel_map,
        k_runs=args.k_runs, n_samples=args.n_samples,
        grid_size=args.grid_size, device=cpu_device,
        device_label="CPU", quadrature_order=args.quadrature_order,
        warmup_runs=args.warmup_runs,
    )
    all_run_results.extend(cpu_results)
    s = _summarize(cpu_results, args.n_samples)
    summaries.append(s)
    print(f"  MAE:        {s['mean_abs_error']:.6f}")
    print(f"  Rel error:  {s['mean_rel_error_pct']:.3f}%")
    print(f"  Throughput: {s['nn_throughput_Msamples_per_sec']:.3f} M samples/s")

    # 5. CIM analytical
    print(f"\nCIM model ({args.array_rows}×{args.array_cols} array, ADC {args.adc_rate:.1e} conv/s)...")
    cim_stats = cim_throughput_estimate(
        args.hidden1, args.hidden2,
        array_rows=args.array_rows, array_cols=args.array_cols,
        adc_rate=args.adc_rate,
        rng_throughput=args.rng_throughput,
        rng_efficiency=args.rng_efficiency,
    )
    print(f"  CIM MACs/sample:    {cim_stats['cim_mac_per_sample']:,}")
    print(f"  ADC ops/sample:     {cim_stats['adc_ops_per_sample']:,}")
    print(f"  RNG-limited:        {cim_stats['rng_limited_Msamples_per_sec']:.3f} M samples/s")
    print(f"  ADC-limited:        {cim_stats['adc_limited_Msamples_per_sec']:.3f} M samples/s")
    print(f"  Bottleneck:         {cim_stats['bottleneck']}")
    print(f"  CIM throughput:     {cim_stats['cim_throughput_Msamples_per_sec']:.3f} M samples/s")
    print(f"  RNG energy/sample:  {cim_stats['rng_energy_per_sample_pJ']:.4f} pJ")

    # 6. Save CSV
    output_path = Path(args.csv)
    write_results(all_run_results, summaries, cim_stats, output_path)
    print(f"\nWrote {output_path}")

    # 7. Histogram — use GPU results if available, else CPU
    plot_results = gpu_results if gpu_results else cpu_results
    plot_histogram(plot_results, Path(args.plot), bins=args.hist_bins)


if __name__ == "__main__":
    main()
