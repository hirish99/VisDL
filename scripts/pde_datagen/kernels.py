"""Green's function kernels for PDE data generation."""
from __future__ import annotations

import math

import torch


class LaplaceKernel2D:
    """2D Laplace Green's function: G(p, s) = -1/(2*pi) * log|p - s|."""

    def evaluate(
        self,
        px: torch.Tensor,   # (B, N)
        py: torch.Tensor,   # (B, N)
        sx: torch.Tensor,   # (B, M)
        sy: torch.Tensor,   # (B, M)
    ) -> torch.Tensor:
        """Returns (B, N, M) kernel matrix."""
        dx = px[:, :, None] - sx[:, None, :]    # (B, N, M)
        dy = py[:, :, None] - sy[:, None, :]
        dist = torch.sqrt(dx * dx + dy * dy)
        return -1.0 / (2.0 * math.pi) * torch.log(dist + 1e-30)


class HelmholtzKernel2D:
    """2D Helmholtz Green's function (real part).

    G(p, s) = -1/4 * Y_0(k|p - s|)

    where Y_0 is the Bessel function of the second kind, order 0.
    """

    def __init__(self, wavenumber: float = 1.0):
        self.k = wavenumber

    def evaluate(
        self,
        px: torch.Tensor,
        py: torch.Tensor,
        sx: torch.Tensor,
        sy: torch.Tensor,
    ) -> torch.Tensor:
        dx = px[:, :, None] - sx[:, None, :]
        dy = py[:, :, None] - sy[:, None, :]
        kr = self.k * torch.sqrt(dx * dx + dy * dy + 1e-30)
        y0 = torch.special.bessel_y0(kr)
        return -0.25 * y0
