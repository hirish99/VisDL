"""Batch pipeline — orchestrates GPU data generation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import TWO_PI
from .protocols import CurveGenerator, SourcePlacer, QuerySampler, Kernel, FeatureAssembler


@dataclass
class PipelineConfig:
    n_samples: int = 1000
    n_sensors: int = 64
    n_queries: int = 100
    min_sources: int = 10
    max_sources: int = 50
    batch_size: int = 1024
    seed: int = 42
    device: str = "cuda"


class BatchPipeline:
    """Orchestrates batched GPU data generation."""

    def __init__(
        self,
        curve_gen: CurveGenerator,
        source_placer: SourcePlacer,
        query_sampler: QuerySampler,
        kernel: Kernel,
        feature_asm: FeatureAssembler,
        config: PipelineConfig,
    ):
        self.curve_gen = curve_gen
        self.source_placer = source_placer
        self.query_sampler = query_sampler
        self.kernel = kernel
        self.feature_asm = feature_asm
        self.config = config

    def generate(self, progress_cb=None) -> torch.Tensor:
        """Generate all data. Returns (n_samples * n_queries, n_cols) on CPU."""
        cfg = self.config
        device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        chunks: list[torch.Tensor] = []
        remaining = cfg.n_samples
        generated = 0

        while remaining > 0:
            B = min(remaining, cfg.batch_size)

            # 1. Valid curves
            curves = self.curve_gen.generate(B, device)

            # 2. Per-curve source count ~ Uniform[min_sources, max_sources]
            n_src = torch.randint(
                cfg.min_sources, cfg.max_sources + 1, (B,), device=device,
            )

            # 3. Sources inside each curve — (B, max_M), zero-padded
            src_x, src_y = self.source_placer.place(curves, n_src, device)
            max_M = src_x.shape[1]

            # 4. Random strengths, masked so unused slots contribute nothing
            mask = torch.arange(max_M, device=device).unsqueeze(0) < n_src.unsqueeze(1)
            strengths = torch.randn(B, max_M, device=device) * mask.float()

            # 5. Query points
            qx, qy = self.query_sampler.sample(curves, cfg.n_queries, device)

            # 6. Green's function at queries → potential
            G_q = self.kernel.evaluate(qx, qy, src_x, src_y)   # (B, N, max_M)
            u = torch.bmm(G_q, strengths.unsqueeze(-1)).squeeze(-1)  # (B, N)

            # 7. Sensor potentials (if needed by assembler)
            sensor_pot = None
            if getattr(self.feature_asm, 'needs_sensors', True):
                t_s = torch.linspace(0, TWO_PI, cfg.n_sensors, device=device)
                sx, sy = curves.eval_xy(t_s)                     # (B, S)
                G_s = self.kernel.evaluate(sx, sy, src_x, src_y)  # (B, S, max_M)
                sensor_pot = torch.bmm(G_s, strengths.unsqueeze(-1)).squeeze(-1)

            # 8. Assemble feature rows
            chunk = self.feature_asm.assemble(
                curves, sensor_pot, qx, qy, u,
                n_sensors=cfg.n_sensors,
            )
            chunks.append(chunk.cpu())

            generated += B
            remaining -= B
            if progress_cb:
                progress_cb(generated, cfg.n_samples)

        return torch.cat(chunks, dim=0)
