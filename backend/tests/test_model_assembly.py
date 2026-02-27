"""Tests for ModelAssembly node: building nn.Sequential from specs."""
import pytest
import torch
import torch.nn as nn

from app.nodes.model_assembly import ModelAssemblyNode, LAYER_BUILDERS


def _device_of(model: nn.Module) -> torch.device:
    """Get the device the model is on."""
    return next(model.parameters()).device


class TestLayerBuilders:
    def test_all_builders_exist(self):
        expected = {"Linear", "ReLU", "Sigmoid", "Tanh", "Dropout", "BatchNorm1d", "Identity"}
        assert expected == set(LAYER_BUILDERS.keys())

    def test_identity_builder(self):
        layer = LAYER_BUILDERS["Identity"]({})
        assert isinstance(layer, nn.Identity)
        x = torch.randn(5, 8)
        assert torch.equal(layer(x), x)


class TestModelAssemblyNormal:
    def test_basic_model(self):
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": 8, "out_features": 1}},
        ]
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs)[0]
        assert isinstance(model, nn.Sequential)
        assert len(model) == 3

        device = _device_of(model)
        x = torch.randn(10, 4, device=device)
        out = model(x)
        assert out.shape == (10, 1)

    def test_with_dataset_infers_input_dim(self):
        specs = [
            {"type": "Linear", "params": {"in_features": None, "out_features": 8}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        dataset = {"X": torch.randn(100, 4), "y": torch.randn(100, 1)}
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs, dataset=dataset)[0]

        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)

    def test_missing_in_features_raises(self):
        specs = [
            {"type": "Linear", "params": {"in_features": None, "out_features": 8}},
        ]
        node = ModelAssemblyNode()
        with pytest.raises(ValueError, match="missing in_features"):
            node.execute(layer_specs=specs)

    def test_missing_batchnorm_num_features_raises(self):
        specs = [
            {"type": "BatchNorm1d", "params": {"num_features": None}},
        ]
        node = ModelAssemblyNode()
        with pytest.raises(ValueError, match="missing num_features"):
            node.execute(layer_specs=specs)


class TestModelAssemblyAblated:
    def test_identity_at_start(self):
        """[Identity, ReLU, Linear(?,1)] with dataset providing input_dim."""
        specs = [
            {"type": "Identity", "params": {}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        dataset = {"X": torch.randn(100, 4), "y": torch.randn(100, 1)}
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs, dataset=dataset)[0]

        assert isinstance(model[0], nn.Identity)
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)

    def test_identity_at_end(self):
        """[Linear(4,8), ReLU, Identity]"""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "ReLU", "params": {}},
            {"type": "Identity", "params": {}},
        ]
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs)[0]

        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 8)  # Output is 8 because Identity doesn't transform

    def test_all_identity(self):
        """Model is pure passthrough — no parameters, stays on CPU."""
        specs = [
            {"type": "Identity", "params": {}},
            {"type": "Identity", "params": {}},
        ]
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs)[0]

        x = torch.randn(5, 4)
        out = model(x)
        assert torch.equal(out, x)  # Pure passthrough

    def test_disabled_dropout(self):
        """Linear -> Identity (instead of Dropout) -> Linear"""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "Identity", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        dataset = {"X": torch.randn(100, 4), "y": torch.randn(100, 1)}
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs, dataset=dataset)[0]

        assert isinstance(model[1], nn.Identity)
        device = _device_of(model)
        x = torch.randn(5, 4, device=device)
        out = model(x)
        assert out.shape == (5, 1)

    def test_disabled_activation_omitted(self):
        """When activation is disabled, it's omitted (not Identity). Chain: [Linear, Linear]."""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs)[0]
        assert len(model) == 2
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.Linear)

    def test_unknown_layer_type_raises(self):
        specs = [{"type": "UnknownLayer", "params": {}}]
        node = ModelAssemblyNode()
        with pytest.raises(ValueError, match="Unknown layer type"):
            node.execute(layer_specs=specs)

    def test_nested_list_specs_flattened(self):
        """ModelAssembly flattens nested lists."""
        specs = [
            [{"type": "Linear", "params": {"in_features": 4, "out_features": 8}}],
            [{"type": "ReLU", "params": {}}],
        ]
        node = ModelAssemblyNode()
        model = node.execute(layer_specs=specs)[0]
        assert len(model) == 2
