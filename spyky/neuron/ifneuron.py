import numpy as np

from typing import Tuple, NoReturn
from spyky.network import NeuronLayer


class IFNeuron(NeuronLayer):
    def __init__(
        self,
        shape: Tuple[int, int],
        v_rest: float = -70.0,
        v_thresh: float = 20.0,
        refractor_period: float = 3,
        v_lower_bound: float = -70.0,
        v_upper_bound: float = 20.0,
    ) -> NoReturn:

        super().__init__(shape)

        self.v_rest = v_rest
        self.v = self.v_rest * np.ones(self.shape)
        self.v_thresh = v_thresh
        self.v_lower_bound = v_lower_bound
        self.v_upper_bound = v_upper_bound

        self.refractor_period = refractor_period
        self.refractor_count = np.zeros(self.shape)

    def tick(self, v_incoming: np.array) -> NoReturn:
        self.v += (self.refractor_count == 0).astype(float) * v_incoming
        self.refractor_count = (self.refractor_count > 0).astype(float) * (
            self.refractor_count - self.dt
        )
        self.spikes = self.v > self.v_thresh

        np.putmask(self.refractor_count, self.spikes, self.refractor_period)
        np.putmask(self.v, self.spikes, self.v_rest)
        np.clip(self.v, self.v_lower_bound, self.v_upper_bound)

        super().tick(v_incoming)

    def reset(self) -> None:
        super().reset()
        self.refractor_count = np.zeros(self.shape)
        self.v = self.v_rest * np.ones(self.shape)
