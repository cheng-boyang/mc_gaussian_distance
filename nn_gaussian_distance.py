"""Neural-network surrogate for 3D Gaussian distance Monte Carlo.

This script trains a small two-layer ReLU network to replace the explicit
Gaussian sampling transform for diagonal 3D Gaussians.

Assumption:
- Each Gaussian has diagonal covariance, represented by its 3D standard
  deviation vector.

Given:
- eps ~ N(0, I_3)
- mu1, std1 for X ~ N(mu1, diag(std1^2))
- mu2, std2 for Y ~ N(mu2, diag(std2^2))

The network learns to output a sample from

    Z = X - Y ~ N(mu1 - mu2, diag(std1^2 + std2^2)).

After training, the script estimates E[||Z||_2] with Monte Carlo using the
network outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import torch
from torch import nn


INPUT_DIM = 21
OUTPUT_DIM = 3


@dataclass(frozen=True)
class MonteCarloSummary:
    """Monte Carlo estimate summary."""

    estimate: float
    variance_estimate: float
    runtime_seconds: float
    num_samples: int


class TwoLayerGaussianSampler(nn.Module):
    """Two-layer ReLU network that maps Gaussian inputs to a 3D sample."""

    def __init__(self, hidden1: int, hidden2: int) -> None:
        super().__init__()
        if hidden1 <= 0 or hidden2 <= 0:
            raise ValueError("hidden layer sizes must be positive.")
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, OUTPUT_DIM),
        )

    def forward(
        self,
        eps: torch.Tensor,
        mu1: torch.Tensor,
        std1: torch.Tensor,
        mu2: torch.Tensor,
        std2: torch.Tensor,
    ) -> torch.Tensor:
        """Return a 3D sample approximation."""
        delta_mu = mu1 - mu2
        reduced_std = torch.sqrt(std1.square() + std2.square())
        features = torch.cat([eps, mu1, std1, mu2, std2, delta_mu, reduced_std], dim=-1)
        return self.net(features)


def parse_vec3(text: str) -> torch.Tensor:
    """Parse a comma-separated 3D vector."""
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got {text!r}.")
    return torch.tensor(values, dtype=torch.float32)


def exact_reduced_sample(
    eps: torch.Tensor,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
) -> torch.Tensor:
    """Exact diagonal-Gaussian reduced sample."""
    return (mu1 - mu2) + eps * torch.sqrt(std1.square() + std2.square())


def sample_parameter_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Sample random training parameters."""
    mu1 = torch.empty(batch_size, 3, device=device).uniform_(-3.0, 3.0)
    mu2 = torch.empty(batch_size, 3, device=device).uniform_(-3.0, 3.0)
    std1 = torch.empty(batch_size, 3, device=device).uniform_(0.05, 3.0)
    std2 = torch.empty(batch_size, 3, device=device).uniform_(0.05, 3.0)
    eps = torch.randn(batch_size, 3, device=device)
    return eps, mu1, std1, mu2, std2


def train_surrogate(
    model: TwoLayerGaussianSampler,
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Train the surrogate to emulate the exact diagonal sampling map."""
    losses: list[float] = []
    model.train()
    params = [param for param in model.parameters() if param.requires_grad]
    first_moments = [torch.zeros_like(param) for param in params]
    second_moments = [torch.zeros_like(param) for param in params]
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8

    for step in range(train_steps):
        eps, mu1, std1, mu2, std2 = sample_parameter_batch(batch_size, device)
        target = exact_reduced_sample(eps, mu1, std1, mu2, std2)
        pred = model(eps, mu1, std1, mu2, std2)
        loss = torch.mean((pred - target) ** 2)

        for param in model.parameters():
            param.grad = None
        loss.backward()
        with torch.no_grad():
            step_index = step + 1
            beta1_correction = 1.0 - beta1**step_index
            beta2_correction = 1.0 - beta2**step_index
            for param, m, v in zip(params, first_moments, second_moments):
                if param.grad is None:
                    continue
                grad = param.grad
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                m_hat = m / beta1_correction
                v_hat = v / beta2_correction
                param.addcdiv_(m_hat, torch.sqrt(v_hat).add_(eps_adam), value=-lr)

        losses.append(float(loss.item()))
        if (step + 1) % max(1, train_steps // 10) == 0:
            print(f"step {step + 1:>6d}/{train_steps}: mse={losses[-1]:.6e}")

    return losses


@torch.no_grad()
def monte_carlo_distance_nn(
    model: TwoLayerGaussianSampler,
    mu1: torch.Tensor,
    std1: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    *,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> MonteCarloSummary:
    """Estimate E[||X - Y||_2] using NN-generated samples."""
    model.eval()
    start = perf_counter()
    sum_norms = torch.zeros((), device=device)
    sum_sq_norms = torch.zeros((), device=device)
    remaining = num_samples

    while remaining > 0:
        current = min(batch_size, remaining)
        eps = torch.randn(current, 3, device=device)
        mu1_batch = mu1.expand(current, -1)
        std1_batch = std1.expand(current, -1)
        mu2_batch = mu2.expand(current, -1)
        std2_batch = std2.expand(current, -1)
        samples = model(eps, mu1_batch, std1_batch, mu2_batch, std2_batch)
        norms = torch.linalg.norm(samples, dim=1)
        sum_norms += norms.sum()
        sum_sq_norms += norms.square().sum()
        remaining -= current

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = perf_counter() - start

    n = float(num_samples)
    mean = sum_norms / n
    second_moment = sum_sq_norms / n
    sample_var = torch.clamp((n / (n - 1.0)) * (second_moment - mean * mean), min=0.0)
    return MonteCarloSummary(
        estimate=float(mean.item()),
        variance_estimate=float((sample_var / n).item()),
        runtime_seconds=runtime,
        num_samples=num_samples,
    )


@torch.no_grad()
def monte_carlo_distance_exact(
    mu1: torch.Tensor,
    std1: torch.Tensor,
    mu2: torch.Tensor,
    std2: torch.Tensor,
    *,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> MonteCarloSummary:
    """Estimate E[||X - Y||_2] using the exact diagonal sampling map."""
    start = perf_counter()
    sum_norms = torch.zeros((), device=device)
    sum_sq_norms = torch.zeros((), device=device)
    remaining = num_samples

    while remaining > 0:
        current = min(batch_size, remaining)
        eps = torch.randn(current, 3, device=device)
        mu1_batch = mu1.expand(current, -1)
        std1_batch = std1.expand(current, -1)
        mu2_batch = mu2.expand(current, -1)
        std2_batch = std2.expand(current, -1)
        samples = exact_reduced_sample(eps, mu1_batch, std1_batch, mu2_batch, std2_batch)
        norms = torch.linalg.norm(samples, dim=1)
        sum_norms += norms.sum()
        sum_sq_norms += norms.square().sum()
        remaining -= current

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = perf_counter() - start

    n = float(num_samples)
    mean = sum_norms / n
    second_moment = sum_sq_norms / n
    sample_var = torch.clamp((n / (n - 1.0)) * (second_moment - mean * mean), min=0.0)
    return MonteCarloSummary(
        estimate=float(mean.item()),
        variance_estimate=float((sample_var / n).item()),
        runtime_seconds=runtime,
        num_samples=num_samples,
    )


def get_device(name: str) -> torch.device:
    """Select the requested device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def print_summary(label: str, summary: MonteCarloSummary) -> None:
    """Print a compact Monte Carlo summary."""
    print(
        f"{label:<14} estimate={summary.estimate:.6f} "
        f"stderr={summary.variance_estimate ** 0.5:.6e} "
        f"time={summary.runtime_seconds:.4f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden1", type=int, default=64, help="First hidden layer width.")
    parser.add_argument("--hidden2", type=int, default=64, help="Second hidden layer width.")
    parser.add_argument("--train-steps", type=int, default=3000, help="Training steps.")
    parser.add_argument("--train-batch-size", type=int, default=4096, help="Training batch size.")
    parser.add_argument("--mc-samples", type=int, default=100000, help="Monte Carlo samples.")
    parser.add_argument("--mc-batch-size", type=int, default=50000, help="Monte Carlo batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cpu, cuda.")
    parser.add_argument("--mu1", default="0,0,0", help="Mean of the first Gaussian.")
    parser.add_argument("--std1", default="1,1,1", help="Standard deviation of the first Gaussian.")
    parser.add_argument("--mu2", default="1,2,3", help="Mean of the second Gaussian.")
    parser.add_argument("--std2", default="1,1,1", help="Standard deviation of the second Gaussian.")
    return parser


def main() -> None:
    """Train the NN surrogate and estimate the Gaussian distance."""
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    mu1 = parse_vec3(args.mu1).to(device).unsqueeze(0)
    std1 = parse_vec3(args.std1).to(device).unsqueeze(0)
    mu2 = parse_vec3(args.mu2).to(device).unsqueeze(0)
    std2 = parse_vec3(args.std2).to(device).unsqueeze(0)

    model = TwoLayerGaussianSampler(args.hidden1, args.hidden2).to(device)

    print(f"device: {device}")
    print(f"hidden layers: {args.hidden1}, {args.hidden2}")
    print("training surrogate...")
    train_surrogate(
        model,
        train_steps=args.train_steps,
        batch_size=args.train_batch_size,
        lr=args.lr,
        device=device,
    )

    print("\nMonte Carlo distance estimation")
    nn_summary = monte_carlo_distance_nn(
        model,
        mu1,
        std1,
        mu2,
        std2,
        num_samples=args.mc_samples,
        batch_size=args.mc_batch_size,
        device=device,
    )
    exact_summary = monte_carlo_distance_exact(
        mu1,
        std1,
        mu2,
        std2,
        num_samples=args.mc_samples,
        batch_size=args.mc_batch_size,
        device=device,
    )

    print_summary("NN surrogate", nn_summary)
    print_summary("Exact diag", exact_summary)
    print(f"abs error:     {abs(nn_summary.estimate - exact_summary.estimate):.6e}")


if __name__ == "__main__":
    main()
