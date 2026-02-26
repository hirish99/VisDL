"""Layer specification nodes: Linear, ReLU, Sigmoid, Tanh, Dropout, BatchNorm1d."""
from typing import Any

from .base import BaseNode, DataType, InputSpec, OutputSpec
from .registry import NodeRegistry


@NodeRegistry.register("Linear")
class LinearNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Linear"
    DESCRIPTION = "Fully connected linear layer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
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
        return [OutputSpec(dtype=DataType.LAYER_SPEC, name="layer_spec")]

    def execute(self, **kwargs) -> tuple:
        spec = {
            "type": "Linear",
            "params": {
                "in_features": kwargs.get("in_features"),
                "out_features": kwargs["out_features"],
                "bias": kwargs.get("bias", True),
            },
        }
        return (spec,)

    def on_disable(self, **kwargs) -> tuple:
        return ({"type": "Identity", "params": {}},)


@NodeRegistry.register("ReLU")
class ReLUNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "ReLU"
    DESCRIPTION = "ReLU activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPEC, name="layer_spec")]

    def execute(self, **kwargs) -> tuple:
        return ({"type": "ReLU", "params": {}},)

    def on_disable(self, **kwargs) -> tuple:
        return ({"type": "Identity", "params": {}},)


@NodeRegistry.register("Sigmoid")
class SigmoidNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Sigmoid"
    DESCRIPTION = "Sigmoid activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPEC, name="layer_spec")]

    def execute(self, **kwargs) -> tuple:
        return ({"type": "Sigmoid", "params": {}},)

    def on_disable(self, **kwargs) -> tuple:
        return ({"type": "Identity", "params": {}},)


@NodeRegistry.register("Tanh")
class TanhNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Tanh"
    DESCRIPTION = "Tanh activation function"

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPEC, name="layer_spec")]

    def execute(self, **kwargs) -> tuple:
        return ({"type": "Tanh", "params": {}},)

    def on_disable(self, **kwargs) -> tuple:
        return ({"type": "Identity", "params": {}},)


@NodeRegistry.register("Dropout")
class DropoutNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "Dropout"
    DESCRIPTION = "Dropout regularization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "p": InputSpec(
                dtype=DataType.FLOAT, default=0.5, required=False,
                min_val=0.0, max_val=1.0, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPEC, name="layer_spec")]

    def execute(self, **kwargs) -> tuple:
        return ({"type": "Dropout", "params": {"p": kwargs.get("p", 0.5)}},)

    def on_disable(self, **kwargs) -> tuple:
        return ({"type": "Identity", "params": {}},)


@NodeRegistry.register("BatchNorm1d")
class BatchNorm1dNode(BaseNode):
    CATEGORY = "Layers"
    DISPLAY_NAME = "BatchNorm1d"
    DESCRIPTION = "1D batch normalization"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "num_features": InputSpec(
                dtype=DataType.INT, default=None, required=False,
                min_val=1, is_handle=False,
            ),
        }

    @classmethod
    def RETURN_TYPES(cls):
        return [OutputSpec(dtype=DataType.LAYER_SPEC, name="layer_spec")]

    def execute(self, **kwargs) -> tuple:
        return ({"type": "BatchNorm1d", "params": {"num_features": kwargs.get("num_features")}},)

    def on_disable(self, **kwargs) -> tuple:
        return ({"type": "Identity", "params": {}},)
