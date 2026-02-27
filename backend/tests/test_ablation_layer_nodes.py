"""Tests for layer node on_disable (ablation) methods."""
from app.nodes.layers import (
    LinearNode, ReLUNode, SigmoidNode, TanhNode,
    DropoutNode, BatchNorm1dNode, _chain,
)


class TestChainHelper:
    def test_chain_none_prev(self):
        spec = {"type": "Linear", "params": {}}
        result = _chain(None, spec)
        assert result == [spec]

    def test_chain_list_prev(self):
        prev = [{"type": "Linear", "params": {}}]
        spec = {"type": "ReLU", "params": {}}
        result = _chain(prev, spec)
        assert result == [{"type": "Linear", "params": {}}, {"type": "ReLU", "params": {}}]

    def test_chain_single_dict_prev(self):
        prev = {"type": "Linear", "params": {}}
        spec = {"type": "ReLU", "params": {}}
        result = _chain(prev, spec)
        assert result == [prev, spec]


class TestLinearNodeAblation:
    def test_on_disable_no_prev(self):
        node = LinearNode()
        result = node.on_disable()
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_on_disable_with_prev(self):
        node = LinearNode()
        prev = [{"type": "Linear", "params": {"in_features": 4, "out_features": 8}}]
        result = node.on_disable(prev_specs=prev)
        specs = result[0]
        assert len(specs) == 2
        assert specs[0] == prev[0]
        assert specs[1] == {"type": "Identity", "params": {}}

    def test_on_disable_with_prev_none_explicit(self):
        node = LinearNode()
        result = node.on_disable(prev_specs=None)
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_execute_vs_disable(self):
        """Executing produces Linear spec, disabling produces Identity."""
        node = LinearNode()
        exec_result = node.execute(out_features=64)
        disable_result = node.on_disable()
        assert exec_result[0][0]["type"] == "Linear"
        assert disable_result[0][0]["type"] == "Identity"


class TestReLUNodeAblation:
    def test_on_disable_no_prev(self):
        node = ReLUNode()
        result = node.on_disable()
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_on_disable_passes_through_prev(self):
        node = ReLUNode()
        prev = [{"type": "Linear", "params": {"in_features": 4, "out_features": 8}}]
        result = node.on_disable(prev_specs=prev)
        assert result == (prev,)
        # Key: activation on_disable does NOT insert Identity, it passes through

    def test_on_disable_does_not_modify_chain(self):
        """Disabling ReLU should remove it from the chain, not add Identity."""
        node = ReLUNode()
        prev = [
            {"type": "Linear", "params": {"in_features": 4, "out_features": 8}},
            {"type": "ReLU", "params": {}},
        ]
        result = node.on_disable(prev_specs=prev)
        # Returns prev unchanged — ReLU is just omitted
        assert result[0] == prev


class TestSigmoidNodeAblation:
    def test_on_disable_no_prev(self):
        node = SigmoidNode()
        result = node.on_disable()
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_on_disable_passes_through(self):
        node = SigmoidNode()
        prev = [{"type": "Linear", "params": {}}]
        result = node.on_disable(prev_specs=prev)
        assert result == (prev,)


class TestTanhNodeAblation:
    def test_on_disable_no_prev(self):
        node = TanhNode()
        result = node.on_disable()
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_on_disable_passes_through(self):
        node = TanhNode()
        prev = [{"type": "Linear", "params": {}}]
        result = node.on_disable(prev_specs=prev)
        assert result == (prev,)


class TestDropoutNodeAblation:
    def test_on_disable_no_prev(self):
        node = DropoutNode()
        result = node.on_disable()
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_on_disable_passes_through(self):
        node = DropoutNode()
        prev = [{"type": "Linear", "params": {}}, {"type": "ReLU", "params": {}}]
        result = node.on_disable(prev_specs=prev)
        assert result == (prev,)

    def test_on_disable_ignores_p_param(self):
        """Dropout's p parameter should be irrelevant when disabled."""
        node = DropoutNode()
        prev = [{"type": "Linear", "params": {}}]
        result = node.on_disable(prev_specs=prev, p=0.9)
        assert result == (prev,)


class TestBatchNorm1dNodeAblation:
    def test_on_disable_no_prev(self):
        node = BatchNorm1dNode()
        result = node.on_disable()
        assert result == ([{"type": "Identity", "params": {}}],)

    def test_on_disable_passes_through(self):
        node = BatchNorm1dNode()
        prev = [{"type": "Linear", "params": {"in_features": 4, "out_features": 8}}]
        result = node.on_disable(prev_specs=prev)
        assert result == (prev,)


class TestLayerAblationChaining:
    """Test ablation behavior in multi-layer chains."""

    def test_disabled_middle_activation(self):
        """Linear -> [disabled ReLU] -> Linear: ReLU just passes through."""
        linear1 = LinearNode()
        relu = ReLUNode()
        linear2 = LinearNode()

        specs = linear1.execute(in_features=4, out_features=8)[0]
        specs = relu.on_disable(prev_specs=specs)[0]  # disabled
        specs = linear2.execute(prev_specs=specs, out_features=1)[0]

        assert len(specs) == 2
        assert specs[0]["type"] == "Linear"
        assert specs[1]["type"] == "Linear"
        # No ReLU in chain

    def test_disabled_first_linear(self):
        """[disabled Linear] -> ReLU -> Linear"""
        linear1 = LinearNode()
        relu = ReLUNode()
        linear2 = LinearNode()

        specs = linear1.on_disable()[0]  # disabled, no prev
        specs = relu.execute(prev_specs=specs)[0]
        specs = linear2.execute(prev_specs=specs, out_features=1)[0]

        assert len(specs) == 3
        assert specs[0]["type"] == "Identity"
        assert specs[1]["type"] == "ReLU"
        assert specs[2]["type"] == "Linear"

    def test_disabled_last_linear(self):
        """Linear -> ReLU -> [disabled Linear]"""
        linear1 = LinearNode()
        relu = ReLUNode()
        linear2 = LinearNode()

        specs = linear1.execute(in_features=4, out_features=8)[0]
        specs = relu.execute(prev_specs=specs)[0]
        specs = linear2.on_disable(prev_specs=specs)[0]  # disabled

        assert len(specs) == 3
        assert specs[0]["type"] == "Linear"
        assert specs[1]["type"] == "ReLU"
        assert specs[2]["type"] == "Identity"

    def test_all_layers_disabled(self):
        """[disabled Linear] -> [disabled ReLU] -> [disabled Linear]"""
        linear1 = LinearNode()
        relu = ReLUNode()
        linear2 = LinearNode()

        specs = linear1.on_disable()[0]
        specs = relu.on_disable(prev_specs=specs)[0]
        specs = linear2.on_disable(prev_specs=specs)[0]

        # Linear1 -> Identity, ReLU passes through, Linear2 -> Identity
        assert len(specs) == 2
        assert specs[0]["type"] == "Identity"
        assert specs[1]["type"] == "Identity"

    def test_disabled_dropout_in_chain(self):
        """Linear -> [disabled Dropout] -> Linear"""
        linear1 = LinearNode()
        dropout = DropoutNode()
        linear2 = LinearNode()

        specs = linear1.execute(in_features=4, out_features=8)[0]
        specs = dropout.on_disable(prev_specs=specs)[0]  # disabled
        specs = linear2.execute(prev_specs=specs, out_features=1)[0]

        assert len(specs) == 2
        assert specs[0]["type"] == "Linear"
        assert specs[1]["type"] == "Linear"
        # No Dropout in chain

    def test_disabled_batchnorm_in_chain(self):
        """Linear -> [disabled BatchNorm] -> ReLU -> Linear"""
        linear1 = LinearNode()
        bn = BatchNorm1dNode()
        relu = ReLUNode()
        linear2 = LinearNode()

        specs = linear1.execute(in_features=4, out_features=8)[0]
        specs = bn.on_disable(prev_specs=specs)[0]  # disabled
        specs = relu.execute(prev_specs=specs)[0]
        specs = linear2.execute(prev_specs=specs, out_features=1)[0]

        types = [s["type"] for s in specs]
        assert types == ["Linear", "ReLU", "Linear"]
