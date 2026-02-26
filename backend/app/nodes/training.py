"""Training loop node with progress callback support."""
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


@NodeRegistry.register("TrainingLoop")
class TrainingLoopNode(BaseNode):
    CATEGORY = "Training"
    DISPLAY_NAME = "Training Loop"
    DESCRIPTION = "Trains the model with given data, loss, and optimizer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "model": InputSpec(dtype=DataType.MODEL, required=True),
            "optimizer": InputSpec(dtype=DataType.OPTIMIZER, required=True),
            "loss_fn": InputSpec(dtype=DataType.LOSS_FN, required=True),
            "train_loader": InputSpec(dtype=DataType.DATASET, required=True),
            "val_loader": InputSpec(dtype=DataType.DATASET, required=False),
            "epochs": InputSpec(
                dtype=DataType.INT, default=10, required=False,
                min_val=1, max_val=10000, is_handle=False,
            ),
            "progress_callback": InputSpec(
                dtype=DataType.ANY, required=False, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.TRAINING_RESULT, name="result")]

    def execute(self, **kwargs) -> tuple:
        model: nn.Module = kwargs["model"]
        optimizer = kwargs["optimizer"]
        loss_fn = kwargs["loss_fn"]
        train_loader = kwargs["train_loader"]
        val_loader = kwargs.get("val_loader")
        epochs = kwargs.get("epochs", 10)
        progress_cb: Callable | None = kwargs.get("progress_callback")

        # Use same device as model
        device = next(model.parameters()).device

        history = {"train_loss": [], "val_loss": [], "epoch": []}

        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)
            history["epoch"].append(epoch + 1)

            # Validation phase
            avg_val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss_sum = 0.0
                n_val = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        pred = model(X_batch)
                        loss = loss_fn(pred, y_batch)
                        val_loss_sum += loss.item()
                        n_val += 1
                avg_val_loss = val_loss_sum / max(n_val, 1)

            history["val_loss"].append(avg_val_loss)

            if progress_cb:
                progress_cb({
                    "type": "training_progress",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                })

        return ({
            "model": model,
            "history": history,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        },)
