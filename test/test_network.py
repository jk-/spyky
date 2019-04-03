from spyky.network import Network
from spyky.network import Connection
from spyky.network.layer import NeuronLayer


class TestNetwork:
    def test_add_layer(self):
        network = Network(1.0)
        layer = NeuronLayer(shape=(1, 1))
        network.add_layer(layer, "layer_1")

        assert network.layers["layer_1"] == layer
        assert layer.network == network
        assert layer.dt == network.dt

    def test_add_connection(self):
        network = Network()
        source = NeuronLayer(shape=(1, 1))
        target = NeuronLayer(shape=(1, 1))
        connection = Connection(source, target)
        network.add_connection(connection, "source", "target")

        assert network.connections[("source", "target")] == connection
        assert connection.network == network
        assert connection.dt == network.dt
