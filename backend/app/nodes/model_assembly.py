"""Model assembly node: chains layer specs into nn.Sequential."""
from typing import Any

import torch
import torch.nn as nn

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


LAYER_BUILDERS: dict[str, Any] = {
    "Linear": lambda p: nn.Linear(p["in_features"], p["out_features"], bias=p.get("bias", True)),
    "ReLU": lambda p: nn.ReLU(),
    "Sigmoid": lambda p: nn.Sigmoid(),
    "Tanh": lambda p: nn.Tanh(),
    "Dropout": lambda p: nn.Dropout(p=p.get("p", 0.5)),
    "BatchNorm1d": lambda p: nn.BatchNorm1d(p["num_features"]),
    "Identity": lambda p: nn.Identity(),
}


def _infer_shapes(specs: list[dict], input_dim: int | None = None) -> list[dict]:
    """Auto-fill in_features/num_features from previous layer's out_features."""
    last_out = input_dim
    for spec in specs:
        params = spec["params"]
        layer_type = spec["type"]

        if layer_type == "Linear":
            if params.get("in_features") is None and last_out is not None:
                params["in_features"] = last_out
            last_out = params.get("out_features", last_out)

        elif layer_type == "BatchNorm1d":
            if params.get("num_features") is None and last_out is not None:
                params["num_features"] = last_out

    return specs


@NodeRegistry.register("ModelAssembly")
class ModelAssemblyNode(BaseNode):
    CATEGORY = "Model"
    DISPLAY_NAME = "Model Assembly"
    DESCRIPTION = "Assembles layer specs into a PyTorch nn.Sequential model. Connect a dataset to auto-infer input dimensions."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "layer_specs": InputSpec(dtype=DataType.LAYER_SPECS, required=True),
            "dataset": InputSpec(dtype=DataType.DATASET, required=False),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.MODEL, name="model")]

    def execute(self, **kwargs) -> tuple:
        specs = kwargs["layer_specs"]
        if not isinstance(specs, list):
            specs = [specs]
        # Flatten in case of nested lists (multiple chains connected)
        flat: list[dict] = []
        for s in specs:
            if isinstance(s, list):
                flat.extend(s)
            else:
                flat.append(s)
        specs = flat

        # Infer input dimension from dataset if provided
        input_dim = None
        dataset = kwargs.get("dataset")
        if dataset is not None:
            X = dataset.get("X") if isinstance(dataset, dict) else None
            if X is not None and hasattr(X, 'shape') and len(X.shape) >= 2:
                input_dim = X.shape[1]

        specs = _infer_shapes(specs, input_dim=input_dim)
        layers: list[nn.Module] = []

        for i, spec in enumerate(specs):
            builder = LAYER_BUILDERS.get(spec["type"])
            if builder is None:
                raise ValueError(f"Unknown layer type: {spec['type']}")
            if spec["type"] == "Linear" and not spec["params"].get("in_features"):
                raise ValueError(
                    f"Linear layer {i} is missing in_features. "
                    f"Connect a dataset to Model Assembly or set it manually."
                )
            if spec["type"] == "BatchNorm1d" and not spec["params"].get("num_features"):
                raise ValueError(
                    f"BatchNorm1d layer {i} is missing num_features. "
                    f"Set it manually or place it after a Linear node."
                )
            layers.append(builder(spec["params"]))

        model = nn.Sequential(*layers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return (model,)
