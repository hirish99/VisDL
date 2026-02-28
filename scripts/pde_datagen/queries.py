"""Query samplers — sample evaluation points in the domain."""
from __future__ import annotations

import torch

from . import TWO_PI
from .types import CurveBatch


class NormalOffsetQuerySampler:
    """Exterior query points along outward normals with exponential offset.

    Produces near-field-biased samples: most queries are close to the boundary.
    """

    def __init__(self, scale: float = 0.5, min_offset: float = 0.05):
        self.scale = scale
        self.min_offset = min_offset

    def sample(
        self,
        curves: CurveBatch,
        n_queries: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (qx, qy) each of shape (B, n_queries)."""
        B = curves.batch_size
        t_q = torch.rand(B, n_queries, device=device) * TWO_PI

        x_bdy, y_bdy = curves.eval_xy_batched(t_q)
        _, _, _, nx, ny = curves.eval_derivatives_batched(t_q)

        offset = torch.empty(B, n_queries, device=device).exponential_(1.0 / self.scale)
        offset = offset + self.min_offset

        qx = x_bdy + offset * nx
        qy = y_bdy + offset * ny
        return qx, qy
