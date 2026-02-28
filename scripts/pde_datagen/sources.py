"""Source placers — put MFS source points inside curves on GPU."""
from __future__ import annotations

import torch

from . import TWO_PI
from .types import CurveBatch


class RejectionSourcePlacer:
    """Place MFS sources uniformly inside curves via batched GPU rejection sampling."""

    def __init__(self, margin: float = 0.9):
        self.margin = margin

    def place(
        self,
        curves: CurveBatch,
        n_sources: int | torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (src_x, src_y) each of shape (B, max_sources).

        n_sources: scalar (same count for all curves) or (B,) tensor of
        per-curve counts.  Positions beyond each curve's count are zero.
        """
        B = curves.batch_size

        # Normalise to a (B,) tensor of per-curve targets
        if isinstance(n_sources, int):
            targets = torch.full((B,), n_sources, dtype=torch.long, device=device)
        else:
            targets = n_sources.to(device=device, dtype=torch.long)

        max_n = targets.max().item()

        # Bounding boxes from boundary evaluation
        t_check = torch.linspace(0, TWO_PI, 500, device=device)
        bx, by = curves.eval_xy(t_check)          # (B, 500)
        x_min = bx.min(dim=1).values               # (B,)
        x_max = bx.max(dim=1).values
        y_min = by.min(dim=1).values
        y_max = by.max(dim=1).values

        # Centroid for polar inside test
        cx = bx.mean(dim=1)                         # (B,)
        cy = by.mean(dim=1)

        # Output buffers — zeros for unused slots
        src_x = torch.zeros(B, max_n, device=device)
        src_y = torch.zeros(B, max_n, device=device)
        filled = torch.zeros(B, dtype=torch.long, device=device)

        oversample = max(max_n * 6, 128)

        while (filled < targets).any().item():
            # Uniform candidates in per-curve bounding boxes
            u = torch.rand(B, oversample, device=device)
            v = torch.rand(B, oversample, device=device)
            cand_x = x_min[:, None] + u * (x_max - x_min)[:, None]
            cand_y = y_min[:, None] + v * (y_max - y_min)[:, None]

            # Polar inside test relative to centroid
            dx = cand_x - cx[:, None]
            dy = cand_y - cy[:, None]
            theta = torch.atan2(dy, dx) % TWO_PI    # (B, oversample)
            r_cand = torch.sqrt(dx * dx + dy * dy)

            r_bdy = curves.eval_r_batched(theta)     # (B, oversample)
            inside = r_cand < r_bdy * self.margin    # (B, oversample)

            # Scatter valid points into output buffers
            for b in range(B):
                target_b = targets[b].item()
                n_filled = filled[b].item()
                if n_filled >= target_b:
                    continue
                valid_idx = inside[b].nonzero(as_tuple=True)[0]
                n_take = min(len(valid_idx), target_b - n_filled)
                if n_take > 0:
                    src_x[b, n_filled:n_filled + n_take] = cand_x[b, valid_idx[:n_take]]
                    src_y[b, n_filled:n_filled + n_take] = cand_y[b, valid_idx[:n_take]]
                    filled[b] += n_take

        return src_x, src_y
