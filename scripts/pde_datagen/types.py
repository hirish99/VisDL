"""Core data structures for batched curve representation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import TWO_PI


@dataclass
class CurveBatch:
    """Batched Fourier star-curve parameterization.

    r(t) = r0 + sum_k [alpha_k cos(kt) + beta_k sin(kt)]

    All tensors on the same device.
    """
    r0: torch.Tensor       # (B,)
    alphas: torch.Tensor   # (B, K)
    betas: torch.Tensor    # (B, K)

    @property
    def batch_size(self) -> int:
        return self.r0.shape[0]

    @property
    def K(self) -> int:
        return self.alphas.shape[1]

    @property
    def device(self) -> torch.device:
        return self.r0.device

    # ------------------------------------------------------------------
    # Evaluation (shared t for all curves)
    # ------------------------------------------------------------------

    def _k_vals(self) -> torch.Tensor:
        """[1, 2, ..., K] on same device."""
        return torch.arange(1, self.K + 1, device=self.device, dtype=self.r0.dtype)

    def eval_r(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate r(t) for all curves.  t: (T,) → returns (B, T).

        Uses matmul: no Python loop over Fourier modes.
        """
        k = self._k_vals()                        # (K,)
        kt = k[:, None] * t[None, :]              # (K, T)
        cos_kt = torch.cos(kt)                    # (K, T)
        sin_kt = torch.sin(kt)                    # (K, T)
        # (B, K) @ (K, T) → (B, T)
        return self.r0[:, None] + self.alphas @ cos_kt + self.betas @ sin_kt

    def eval_xy(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cartesian boundary positions.  t: (T,) → (B, T) each."""
        r = self.eval_r(t)
        return r * torch.cos(t), r * torch.sin(t)

    def eval_derivatives(
        self, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tangent, curvature, and outward unit normal.

        t: (T,) → returns (dx/dt, dy/dt, kappa, nx, ny) each (B, T).
        """
        k = self._k_vals()                        # (K,)
        kt = k[:, None] * t[None, :]              # (K, T)
        cos_kt = torch.cos(kt)
        sin_kt = torch.sin(kt)

        r = self.r0[:, None] + self.alphas @ cos_kt + self.betas @ sin_kt   # (B, T)

        # dr/dt = sum_k [-alpha_k * k * sin(kt) + beta_k * k * cos(kt)]
        k_a = (self.alphas * k[None, :])           # (B, K)  — alpha_k * k
        k_b = (self.betas * k[None, :])            # (B, K)
        dr = -k_a @ sin_kt + k_b @ cos_kt          # (B, T)

        # d2r/dt2
        k2_a = (self.alphas * (k ** 2)[None, :])   # (B, K)
        k2_b = (self.betas * (k ** 2)[None, :])    # (B, K)
        d2r = -k2_a @ cos_kt - k2_b @ sin_kt       # (B, T)

        cost = torch.cos(t)
        sint = torch.sin(t)
        dx = dr * cost - r * sint
        dy = dr * sint + r * cost
        d2x = d2r * cost - 2 * dr * sint - r * cost
        d2y = d2r * sint + 2 * dr * cost - r * sint

        speed_sq = dx * dx + dy * dy
        speed = torch.sqrt(speed_sq)
        eps = 1e-30
        kappa = (dx * d2y - dy * d2x) / (speed_sq * speed + eps)

        nx = dy / (speed + eps)
        ny = -dx / (speed + eps)

        return dx, dy, kappa, nx, ny

    # ------------------------------------------------------------------
    # Evaluation (per-curve t: shape (B, T))
    # ------------------------------------------------------------------

    def eval_r_batched(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate r(t) where t is (B, T) — different angles per curve.

        Returns (B, T).
        """
        k = self._k_vals()                            # (K,)
        # kt: (B, K, T) = k[None, :, None] * t[:, None, :]
        kt = k[None, :, None] * t[:, None, :]
        cos_kt = torch.cos(kt)                        # (B, K, T)
        sin_kt = torch.sin(kt)                        # (B, K, T)
        # (B, K, 1) * (B, K, T) → sum over K → (B, T)
        r = (self.r0[:, None]
             + (self.alphas[:, :, None] * cos_kt).sum(dim=1)
             + (self.betas[:, :, None] * sin_kt).sum(dim=1))
        return r

    def eval_xy_batched(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cartesian positions with per-curve t.  t: (B, T) → (B, T) each."""
        r = self.eval_r_batched(t)
        return r * torch.cos(t), r * torch.sin(t)

    def eval_derivatives_batched(
        self, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Derivatives with per-curve t.  t: (B, T) → each (B, T)."""
        k = self._k_vals()
        kt = k[None, :, None] * t[:, None, :]         # (B, K, T)
        cos_kt = torch.cos(kt)
        sin_kt = torch.sin(kt)

        r = (self.r0[:, None]
             + (self.alphas[:, :, None] * cos_kt).sum(dim=1)
             + (self.betas[:, :, None] * sin_kt).sum(dim=1))

        k_a = self.alphas * k[None, :]
        k_b = self.betas * k[None, :]
        dr = ((-k_a[:, :, None] * sin_kt).sum(dim=1)
              + (k_b[:, :, None] * cos_kt).sum(dim=1))

        k2_a = self.alphas * (k ** 2)[None, :]
        k2_b = self.betas * (k ** 2)[None, :]
        d2r = ((-k2_a[:, :, None] * cos_kt).sum(dim=1)
               - (k2_b[:, :, None] * sin_kt).sum(dim=1))

        cost = torch.cos(t)
        sint = torch.sin(t)
        dx = dr * cost - r * sint
        dy = dr * sint + r * cost
        d2x = d2r * cost - 2 * dr * sint - r * cost
        d2y = d2r * sint + 2 * dr * cost - r * sint

        speed_sq = dx * dx + dy * dy
        speed = torch.sqrt(speed_sq)
        eps = 1e-30
        kappa = (dx * d2y - dy * d2x) / (speed_sq * speed + eps)
        nx = dy / (speed + eps)
        ny = -dx / (speed + eps)

        return dx, dy, kappa, nx, ny

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perimeter, area, centroid_x, centroid_y — each (B,)."""
        n_pts = 1000
        t = torch.linspace(0, TWO_PI, n_pts, device=self.device, dtype=self.r0.dtype)
        dt = TWO_PI / n_pts

        x, y = self.eval_xy(t)           # (B, n_pts)
        dx, dy, _, _, _ = self.eval_derivatives(t)

        speed = torch.sqrt(dx * dx + dy * dy)
        perimeter = speed.sum(dim=1) * dt                      # (B,)

        x_next = torch.roll(x, -1, dims=1)
        y_next = torch.roll(y, -1, dims=1)
        area = 0.5 * torch.abs((x * y_next - x_next * y).sum(dim=1))  # (B,)

        cx = x.mean(dim=1)
        cy = y.mean(dim=1)

        return perimeter, area, cx, cy

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def slice(self, mask: torch.Tensor) -> CurveBatch:
        """Return subset of curves matching boolean mask."""
        return CurveBatch(
            r0=self.r0[mask],
            alphas=self.alphas[mask],
            betas=self.betas[mask],
        )
