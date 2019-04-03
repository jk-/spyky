import pytest
import numpy as np

from spyky.network import Connection
from spyky.network.layer import NeuronLayer


class TestConnection:
    def setup_method(self):
        pass

    def test_invalid_source_target(self):
        source = {}
        target = {}
        with pytest.raises(AssertionError):
            connection = Connection(source, target)

    def test_weight_clip(self):
        # missing test for no weights, but weight_min, weight_max
        # missing test for weights, but no weight_min, weight_max
        weights = np.array([-0.1, -0.5, 1.2])

        connection = Connection(
            NeuronLayer((1, 1)),
            NeuronLayer((1, 1)),
            weights=weights,
            weight_min=0,
            weight_max=1,
        )
        assert all([a == b for a, b in zip(connection.weights, [0, 0, 1])])

    def test_calculate(self):
        weights = np.array([0, 1, 2])
        bias = np.array([1, 1, 1])

        connection = Connection(
            NeuronLayer((1, 1)),
            NeuronLayer((1, 1)),
            weights=weights,
            bias=bias,
        )
        spikes = np.array([0, 1, 0])
        weights_calc = connection.calculate(spikes)

        assert all([a == b for a, b in zip(weights_calc, [2.0, 2.0, 2.0])])
