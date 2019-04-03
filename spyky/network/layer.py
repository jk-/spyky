import numpy as np
from operator import mul
from functools import reduce
from typing import Tuple, NoReturn
from abc import ABC, abstractmethod


class AbstractLayer(ABC):
    def __init__(self, shape: Tuple[int, int]) -> NoReturn:
        super().__init__()
        self.shape = shape
        self.network = None
        self.dt = None

    @abstractmethod
    def tick(self, v_incoming: np.array) -> NoReturn:
        pass

    @abstractmethod
    def reset(self) -> NoReturn:
        pass


class NeuronLayer(AbstractLayer):
    def __init__(self, shape: Tuple[int, int]) -> NoReturn:
        super().__init__(shape)
        self.spikes = np.zeros(self.shape)
        self.neuron_count = self.spikes.size

    def tick(self, v_incoming: np.array) -> NoReturn:
        super().tick(v_incoming)

    def reset(self) -> NoReturn:
        super().reset()
        self.spikes = np.zeros(self.shape)
