"""Feature assemblers — build per-row output tensors from curve/query data."""
from __future__ import annotations

import torch

from . import TWO_PI
from .types import CurveBatch


class SimpleAssembler:
    """Output: query_x, query_y, u (3 columns per row)."""

    def assemble(
        self,
        curves: CurveBatch,
        sensor_potentials: torch.Tensor | None,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        target_u: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        # (B, Q, 3) → (B*Q, 3)
        return torch.stack([query_x, query_y, target_u], dim=-1).reshape(-1, 3)

    def header(self, n_sensors: int) -> list[str]:
        return ["x", "y", "u"]

    @property
    def needs_sensors(self) -> bool:
        return False


class CompactAssembler:
    """Output: s_0_g, ..., s_N_g, query_x, query_y, u."""

    def assemble(
        self,
        curves: CurveBatch,
        sensor_potentials: torch.Tensor | None,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        target_u: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        assert sensor_potentials is not None
        B, Q = query_x.shape
        S = sensor_potentials.shape[1]
        # Broadcast sensors (B, S) → (B, Q, S)  — zero-copy expand
        sensors = sensor_potentials[:, None, :].expand(B, Q, S)
        # Stack query + target: each (B, Q, 1)
        rows = torch.cat([
            sensors,
            query_x.unsqueeze(-1),
            query_y.unsqueeze(-1),
            target_u.unsqueeze(-1),
        ], dim=-1)
        return rows.reshape(-1, S + 3)

    def header(self, n_sensors: int) -> list[str]:
        cols = [f"s_{i}_g" for i in range(n_sensors)]
        cols.extend(["query_x", "query_y", "u"])
        return cols

    @property
    def needs_sensors(self) -> bool:
        return True


class FullAssembler:
    """Output: all 8 sensor features + 4 globals + query + target.

    Per sensor: x, y, g, dx/dt, dy/dt, curvature, nx, ny
    Globals: perimeter, area, centroid_x, centroid_y
    """

    def assemble(
        self,
        curves: CurveBatch,
        sensor_potentials: torch.Tensor | None,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        target_u: torch.Tensor,
        *,
        n_sensors: int = 64,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        assert sensor_potentials is not None
        B, Q = query_x.shape
        S = sensor_potentials.shape[1]
        device = query_x.device

        t_sensors = torch.linspace(0, TWO_PI, S, device=device)
        sx, sy = curves.eval_xy(t_sensors)                        # (B, S)
        dx_dt, dy_dt, kappa, nx, ny = curves.eval_derivatives(t_sensors)
        perimeter, area, cx, cy = curves.geometry()               # (B,) each

        # Per-sensor features: (B, S, 8)
        sensor_feats = torch.stack([
            sx, sy, sensor_potentials, dx_dt, dy_dt, kappa, nx, ny,
        ], dim=-1)                                                 # (B, S, 8)

        # Flatten sensors: (B, S*8)
        sensor_flat = sensor_feats.reshape(B, S * 8)

        # Global features: (B, 4)
        globals_ = torch.stack([perimeter, area, cx, cy], dim=-1)

        # Combine: (B, S*8 + 4)
        static = torch.cat([sensor_flat, globals_], dim=-1)

        # Broadcast to queries: (B, Q, S*8+4)
        static_exp = static[:, None, :].expand(B, Q, -1)

        # Cat with query coords + target
        rows = torch.cat([
            static_exp,
            query_x.unsqueeze(-1),
            query_y.unsqueeze(-1),
            target_u.unsqueeze(-1),
        ], dim=-1)

        return rows.reshape(-1, S * 8 + 4 + 3)

    def header(self, n_sensors: int) -> list[str]:
        feat_names = ["x", "y", "g", "dx", "dy", "k", "nx", "ny"]
        cols = []
        for i in range(n_sensors):
            for f in feat_names:
                cols.append(f"s_{i}_{f}")
        cols.extend(["perimeter", "area", "centroid_x", "centroid_y"])
        cols.extend(["query_x", "query_y", "u"])
        return cols

    @property
    def needs_sensors(self) -> bool:
        return True
