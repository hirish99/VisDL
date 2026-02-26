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


def _infer_shapes(specs: list[dict]) -> list[dict]:
    """Auto-fill in_features/num_features from previous layer's out_features."""
    last_out = None
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
    DESCRIPTION = "Assembles layer specs into a PyTorch nn.Sequential model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "layer_specs": InputSpec(dtype=DataType.LAYER_SPECS, required=True),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.MODEL, name="model")]

    def execute(self, **kwargs) -> tuple:
        specs = kwargs["layer_specs"]
        if not isinstance(specs, list):
            specs = [specs]

        specs = _infer_shapes(specs)
        layers: list[nn.Module] = []

        for spec in specs:
            builder = LAYER_BUILDERS.get(spec["type"])
            if builder is None:
                raise ValueError(f"Unknown layer type: {spec['type']}")
            layers.append(builder(spec["params"]))

        model = nn.Sequential(*layers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return (model,)
