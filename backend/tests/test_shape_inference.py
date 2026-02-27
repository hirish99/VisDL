"""Tests for _infer_shapes in model_assembly.py."""
import torch

from app.nodes.model_assembly import _infer_shapes


class TestInferShapesNormal:
    def test_basic_linear_chain(self):
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs, input_dim=4)
        assert result[2]["params"]["in_features"] == 8

    def test_input_dim_fills_first_linear(self):
        specs = [
            {"type": "Linear", "params": {"in_features": None, "out_features": 8}},
        ]
        result = _infer_shapes(specs, input_dim=4)
        assert result[0]["params"]["in_features"] == 4

    def test_no_input_dim_leaves_none(self):
        specs = [
            {"type": "Linear", "params": {"in_features": None, "out_features": 8}},
        ]
        result = _infer_shapes(specs, input_dim=None)
        assert result[0]["params"]["in_features"] is None

    def test_batchnorm_after_linear(self):
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "BatchNorm1d", "params": {"num_features": None}},
        ]
        result = _infer_shapes(specs)
        assert result[1]["params"]["num_features"] == 8

    def test_multiple_linears(self):
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 32}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 16}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs)
        assert result[2]["params"]["in_features"] == 32
        assert result[4]["params"]["in_features"] == 16


class TestInferShapesAblated:
    def test_ablated_first_linear_identity(self):
        """Identity at start: last_out comes from input_dim."""
        specs = [
            {"type": "Identity", "params": {}},
            {"type": "ReLU", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs, input_dim=4)
        # Identity doesn't update last_out, so it stays as input_dim=4
        assert result[2]["params"]["in_features"] == 4

    def test_ablated_middle_activation_removed(self):
        """When ReLU is disabled, it's simply omitted from specs."""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            # ReLU omitted (disabled ReLU passes through prev_specs)
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs)
        assert result[1]["params"]["in_features"] == 8

    def test_ablated_last_linear_identity(self):
        """Identity at end: shape passes through from previous layer."""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "ReLU", "params": {}},
            {"type": "Identity", "params": {}},
        ]
        result = _infer_shapes(specs)
        # Identity doesn't change anything, last_out remains 8
        # No more linears to infer, so nothing to check except it doesn't crash

    def test_all_disabled_identity_only(self):
        """All Identity: nothing to infer, should not crash."""
        specs = [
            {"type": "Identity", "params": {}},
            {"type": "Identity", "params": {}},
        ]
        result = _infer_shapes(specs, input_dim=4)
        assert len(result) == 2

    def test_batchnorm_after_identity_uses_input_dim(self):
        """BatchNorm after Identity gets num_features from input_dim."""
        specs = [
            {"type": "Identity", "params": {}},
            {"type": "BatchNorm1d", "params": {"num_features": None}},
        ]
        result = _infer_shapes(specs, input_dim=8)
        assert result[1]["params"]["num_features"] == 8

    def test_batchnorm_after_identity_no_input_dim(self):
        """BatchNorm after Identity with no input_dim leaves None."""
        specs = [
            {"type": "Identity", "params": {}},
            {"type": "BatchNorm1d", "params": {"num_features": None}},
        ]
        result = _infer_shapes(specs, input_dim=None)
        assert result[1]["params"]["num_features"] is None

    def test_linear_after_identity_after_linear(self):
        """Linear(4,8) -> Identity (disabled Linear) -> Linear(?,1)"""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "Identity", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs)
        # Identity doesn't update last_out, so it stays 8
        assert result[2]["params"]["in_features"] == 8

    def test_identity_preserves_last_out_through_chain(self):
        """Multiple identities don't break the shape tracking."""
        specs = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 16}},
            {"type": "Identity", "params": {}},
            {"type": "Identity", "params": {}},
            {"type": "Linear", "params": {"in_features": None, "out_features": 1}},
        ]
        result = _infer_shapes(specs)
        assert result[3]["params"]["in_features"] == 16
