"""Layer specification nodes: Linear, ReLU, Sigmoid, Tanh, Dropout, BatchNorm1d.

Each layer has an optional `prev_specs` input (chain from previous layer)
and a `layer_specs` output (the chain so far including this layer).
This lets you visually chain: Linear → ReLU → Linear → ModelAssembly.
"""
from typing import Any

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


def _chain(prev: Any, spec: dict) -> list[dict]:
    """Append spec to the chain from previous layers."""
    if prev is None:
        return [spec]
    if isinstance(prev, list):
        return prev + [spec]
    return [prev, spec]


@NodeRegistry.register("Linear")
class LinearNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Linear"
    DESCRIPTION = "Fully connected linear layer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "prev_specs": InputSpec(
                dtype=DataType.LAYER_SPECS, required=False, is_handle=True,
            ),
            "in_features": InputSpec(
                dtype=DataType.INT, default=None, required=False,
                min_val=1, is_handle=False,
            ),
            "out_features": InputSpec(
                dtype=DataType.INT, default=64, required=True,
                min_val=1, is_handle=False,
            ),
            "bias": InputSpec(
                dtype=DataType.BOOL, default=True, required=False,
                is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPECS, name="layer_specs")]

    def execute(self, **kwargs) -> tuple:
        spec = {
            "type": "Linear",
            "params": {
                "in_features": kwargs.get("in_features"),
                "out_features": kwargs["out_features"],
                "bias": kwargs.get("bias", True),
            },
        }
        return (_chain(kwargs.get("prev_specs"), spec),)

    def on_disable(self, **kwargs) -> tuple:
        prev = kwargs.get("prev_specs")
        return (_chain(prev, {"type": "Identity", "params": {}}) if prev else [{"type": "Identity", "params": {}}],)


@NodeRegistry.register("ReLU")
class ReLUNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "ReLU"
    DESCRIPTION = "ReLU activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "prev_specs": InputSpec(
                dtype=DataType.LAYER_SPECS, required=False, is_handle=True,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPECS, name="layer_specs")]

    def execute(self, **kwargs) -> tuple:
        return (_chain(kwargs.get("prev_specs"), {"type": "ReLU", "params": {}}),)

    def on_disable(self, **kwargs) -> tuple:
        prev = kwargs.get("prev_specs")
        return (prev if prev else [{"type": "Identity", "params": {}}],)


@NodeRegistry.register("Sigmoid")
class SigmoidNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Sigmoid"
    DESCRIPTION = "Sigmoid activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "prev_specs": InputSpec(
                dtype=DataType.LAYER_SPECS, required=False, is_handle=True,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPECS, name="layer_specs")]

    def execute(self, **kwargs) -> tuple:
        return (_chain(kwargs.get("prev_specs"), {"type": "Sigmoid", "params": {}}),)

    def on_disable(self, **kwargs) -> tuple:
        prev = kwargs.get("prev_specs")
        return (prev if prev else [{"type": "Identity", "params": {}}],)


@NodeRegistry.register("Tanh")
class TanhNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Tanh"
    DESCRIPTION = "Tanh activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "prev_specs": InputSpec(
                dtype=DataType.LAYER_SPECS, required=False, is_handle=True,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPECS, name="layer_specs")]

    def execute(self, **kwargs) -> tuple:
        return (_chain(kwargs.get("prev_specs"), {"type": "Tanh", "params": {}}),)

    def on_disable(self, **kwargs) -> tuple:
        prev = kwargs.get("prev_specs")
        return (prev if prev else [{"type": "Identity", "params": {}}],)


@NodeRegistry.register("Dropout")
class DropoutNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Dropout"
    DESCRIPTION = "Dropout regularization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "prev_specs": InputSpec(
                dtype=DataType.LAYER_SPECS, required=False, is_handle=True,
            ),
            "p": InputSpec(
                dtype=DataType.FLOAT, default=0.5, required=False,
                min_val=0.0, max_val=1.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPECS, name="layer_specs")]

    def execute(self, **kwargs) -> tuple:
        return (_chain(kwargs.get("prev_specs"), {"type": "Dropout", "params": {"p": kwargs.get("p", 0.5)}}),)

    def on_disable(self, **kwargs) -> tuple:
        prev = kwargs.get("prev_specs")
        return (prev if prev else [{"type": "Identity", "params": {}}],)


@NodeRegistry.register("BatchNorm1d")
class BatchNorm1dNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "BatchNorm1d"
    DESCRIPTION = "1D batch normalization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "prev_specs": InputSpec(
                dtype=DataType.LAYER_SPECS, required=False, is_handle=True,
            ),
            "num_features": InputSpec(
                dtype=DataType.INT, default=None, required=False,
                min_val=1, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPECS, name="layer_specs")]

    def execute(self, **kwargs) -> tuple:
        return (_chain(kwargs.get("prev_specs"), {"type": "BatchNorm1d", "params": {"num_features": kwargs.get("num_features")}}),)

    def on_disable(self, **kwargs) -> tuple:
        prev = kwargs.get("prev_specs")
        return (prev if prev else [{"type": "Identity", "params": {}}],)
