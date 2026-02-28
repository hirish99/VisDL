"""Output writers for generated data."""
from __future__ import annotations

import csv
from pathlib import Path

import torch


def write_csv(tensor: torch.Tensor, header: list[str], path: Path) -> None:
    """Write tensor to CSV. Streams in 10K-row chunks to limit memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = tensor.numpy()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        chunk_size = 10_000
        for start in range(0, len(data), chunk_size):
            writer.writerows(data[start:start + chunk_size].tolist())
