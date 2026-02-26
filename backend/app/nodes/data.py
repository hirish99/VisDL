"""Data nodes: CSV loading and train/val splitting."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry
from ..config import settings


@NodeRegistry.register("CSVLoader")
class CSVLoaderNode(BaseNode):
    CATEGORY = "Data"
    DISPLAY_NAME = "CSV Loader"
    DESCRIPTION = "Load a CSV file and select input/target columns"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "file_id": InputSpec(dtype=DataType.STRING, required=True, is_handle=False),
            "input_columns": InputSpec(
                dtype=DataType.STRING, required=True, is_handle=False,
                default="",
            ),
            "target_columns": InputSpec(
                dtype=DataType.STRING, required=True, is_handle=False,
                default="",
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.DATASET, name="dataset")]

    def execute(self, **kwargs) -> tuple:
        file_id = kwargs["file_id"]
        file_path = settings.upload_dir / file_id
        if not file_path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {file_id}")

        df = pd.read_csv(file_path)

        input_cols = [c.strip() for c in kwargs["input_columns"].split(",") if c.strip()]
        target_cols = [c.strip() for c in kwargs["target_columns"].split(",") if c.strip()]

        if not input_cols:
            raise ValueError("No input columns specified")
        if not target_cols:
            raise ValueError("No target columns specified")

        X = torch.tensor(df[input_cols].values, dtype=torch.float32)
        y = torch.tensor(df[target_cols].values, dtype=torch.float32)

        return ({"X": X, "y": y, "columns": {"input": input_cols, "target": target_cols}},)


@NodeRegistry.register("DataSplitter")
class DataSplitterNode(BaseNode):
    CATEGORY = "Data"
    DISPLAY_NAME = "Data Splitter"
    DESCRIPTION = "Split dataset into train and validation sets"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "dataset": InputSpec(dtype=DataType.DATASET, required=True),
            "val_ratio": InputSpec(
                dtype=DataType.FLOAT, default=0.2, required=False,
                min_val=0.01, max_val=0.99, is_handle=False,
            ),
            "batch_size": InputSpec(
                dtype=DataType.INT, default=32, required=False,
                min_val=1, is_handle=False,
            ),
            "shuffle": InputSpec(
                dtype=DataType.BOOL, default=True, required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [
            OutputSpec(dtype=DataType.DATASET, name="train_loader"),
            OutputSpec(dtype=DataType.DATASET, name="val_loader"),
        ]

    def execute(self, **kwargs) -> tuple:
        dataset = kwargs["dataset"]
        val_ratio = kwargs.get("val_ratio", 0.2)
        batch_size = kwargs.get("batch_size", 32)
        shuffle = kwargs.get("shuffle", True)

        X, y = dataset["X"], dataset["y"]
        n = len(X)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val

        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_ds = TensorDataset(X[train_idx], y[train_idx])
        val_ds = TensorDataset(X[val_idx], y[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return (train_loader, val_loader)
