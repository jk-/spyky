import numpy as np
from typing import Dict, NoReturn
from spyky.network import AbstractLayer
from spyky.network import AbstractConnection
from spyky.network import AbstractProbe


class Network:
    def __init__(self, dt: float = 1.0) -> NoReturn:
        self.dt = dt
        self.layers = {}
        self.connections = {}
        self.probes = {}

    def add_layer(self, layer: AbstractLayer, name: str) -> NoReturn:
        self.layers[name] = layer
        layer.network = self
        layer.dt = self.dt

    def add_connection(
        self, connection: AbstractConnection, source: str, target: str
    ) -> NoReturn:
        self.connections[(source, target)] = connection
        connection.network = self
        connection.dt = self.dt

    def add_probe(self, probe: AbstractProbe, name: str) -> NoReturn:
        self.probes[name] = probe
        probe.network = self
        probe.dt = self.dt

    def get_inputs(self) -> Dict[str, np.array]:
        inpts = {}

        for conn in self.connections:
            source = self.connections[conn].source
            target = self.connections[conn].target

            if not conn[1] in inpts:
                inpts[conn[1]] = np.zeros(target.shape)

            # print("conn", conn[1], source.spikes)
            inpts[conn[1]] += self.connections[conn].calculate(source.spikes)
            # print("got inputs", inpts)

        return inpts

    def run(
        self, inpts: Dict[str, np.array], time: int = 100, **kwargs
    ) -> NoReturn:

        timesteps = int(time / self.dt)

        inpts.update(self.get_inputs())

        for t in range(timesteps):
            for l in self.layers:
                self.layers[l].tick(v_incoming=inpts[l])

            for c in self.connections:
                self.connections[c].update(**kwargs)

            inpts.update(self.get_inputs())

            for m in self.probes:
                self.probes[m].save()
