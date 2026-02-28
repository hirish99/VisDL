"""Curve generators — produce batches of valid closed curves on GPU."""
from __future__ import annotations

import torch

from . import TWO_PI
from .types import CurveBatch


class FourierStarCurveGenerator:
    """Random star-shaped curves via Fourier parameterization.

    r(t) = r0 + sum_k [alpha_k cos(kt) + beta_k sin(kt)]
    with 1/k^2 spectral decay. Curves with r(t) <= min_r anywhere are rejected.
    """

    DIFFICULTY = {
        "easy": (0.05, 3),
        "medium": (0.15, 5),
        "hard": (0.25, 8),
    }

    def __init__(self, difficulty: str = "medium", min_r: float = 0.05):
        self.difficulty = difficulty
        self.min_r = min_r
        self.sigma, self.K = self.DIFFICULTY[difficulty]

    def generate(self, batch_size: int, device: torch.device) -> CurveBatch:
        """Generate exactly batch_size valid curves via oversampling + rejection."""
        t_check = torch.linspace(0, TWO_PI, 500, device=device)
        k_vals = torch.arange(1, self.K + 1, device=device, dtype=torch.float32)
        mode_sigmas = self.sigma / (k_vals ** 2)    # (K,)

        parts_r0: list[torch.Tensor] = []
        parts_a: list[torch.Tensor] = []
        parts_b: list[torch.Tensor] = []
        collected = 0
        overshoot = 1.5

        while collected < batch_size:
            needed = batch_size - collected
            gen = int(needed * overshoot) + 16

            r0 = torch.empty(gen, device=device).uniform_(0.8, 1.5)
            alphas = torch.randn(gen, self.K, device=device) * mode_sigmas
            betas = torch.randn(gen, self.K, device=device) * mode_sigmas

            batch = CurveBatch(r0=r0, alphas=alphas, betas=betas)
            r_vals = batch.eval_r(t_check)              # (gen, 500)
            valid = r_vals.min(dim=1).values > self.min_r

            if valid.any():
                parts_r0.append(r0[valid])
                parts_a.append(alphas[valid])
                parts_b.append(betas[valid])
                collected += int(valid.sum().item())

            accept = valid.float().mean().item()
            if accept > 0:
                overshoot = min(1.0 / accept * 1.1, 10.0)

        return CurveBatch(
            r0=torch.cat(parts_r0)[:batch_size],
            alphas=torch.cat(parts_a)[:batch_size],
            betas=torch.cat(parts_b)[:batch_size],
        )
