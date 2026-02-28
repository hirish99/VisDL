"""Protocol definitions for swappable data generation components."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .types import CurveBatch


@runtime_checkable
class CurveGenerator(Protocol):
    """Generates batches of valid closed curves on GPU."""

    def generate(self, batch_size: int, device: torch.device) -> CurveBatch: ...


@runtime_checkable
class SourcePlacer(Protocol):
    """Places source points inside curves."""

    def place(
        self,
        curves: CurveBatch,
        n_sources: int | torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (src_x, src_y) each of shape (B, max_sources).

        n_sources can be a scalar (same for all curves) or a (B,) tensor
        of per-curve counts.  Unused slots are zero-padded.
        """
        ...


@runtime_checkable
class QuerySampler(Protocol):
    """Samples query points in the domain of interest."""

    def sample(
        self,
        curves: CurveBatch,
        n_queries: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (qx, qy) each of shape (B, n_queries)."""
        ...


@runtime_checkable
class Kernel(Protocol):
    """Green's function or fundamental solution kernel."""

    def evaluate(
        self,
        px: torch.Tensor,   # (B, N)
        py: torch.Tensor,   # (B, N)
        sx: torch.Tensor,   # (B, M)
        sy: torch.Tensor,   # (B, M)
    ) -> torch.Tensor:
        """Returns kernel matrix of shape (B, N, M)."""
        ...


@runtime_checkable
class FeatureAssembler(Protocol):
    """Assembles per-row feature tensors from curve/query/potential data."""

    def assemble(
        self,
        curves: CurveBatch,
        sensor_potentials: torch.Tensor | None,  # (B, n_sensors) or None
        query_x: torch.Tensor,                   # (B, n_queries)
        query_y: torch.Tensor,                   # (B, n_queries)
        target_u: torch.Tensor,                  # (B, n_queries)
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Returns feature tensor of shape (B * n_queries, n_cols)."""
        ...

    def header(self, n_sensors: int) -> list[str]:
        """Column names for the output CSV."""
        ...
