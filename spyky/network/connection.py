import numpy as np

from abc import ABC, abstractmethod
from typing import NoReturn
from spyky.network import AbstractLayer


class AbstractConnection(ABC):
    def __init__(
        self, source: AbstractLayer, target: AbstractLayer, **kwargs
    ) -> NoReturn:
        assert isinstance(
            source, AbstractLayer
        ), "Source is not of type AbstractLayer"
        assert isinstance(
            target, AbstractLayer
        ), "Target is not of type AbstractLayer"

        super().__init__()
        self.source = source
        self.target = target
        self.weights = None

        self.weight_min = kwargs.get("weight_min", -np.inf)
        self.weight_max = kwargs.get("weight_max", np.inf)

    @abstractmethod
    def calculate(self, spikes: np.array) -> NoReturn:
        pass

    @abstractmethod
    def update(self, **kwargs) -> NoReturn:
        pass

    @abstractmethod
    def reset(self) -> NoReturn:
        pass


class Connection(AbstractConnection):
    def __init__(
        self, source: AbstractLayer, target: AbstractLayer, **kwargs
    ) -> NoReturn:
        super().__init__(source, target, **kwargs)

        self.weights = kwargs.get("weights", None)
        self.bias = kwargs.get("bias", np.zeros(target.neuron_count))

        if self.weights is None:
            if self.weight_min == -np.inf or self.weight_max == np.inf:
                self.weights = np.clip(
                    np.random.random(
                        (source.neuron_count, target.neuron_count)
                    ),
                    self.weight_min,
                    self.weight_max,
                )
            else:
                self.weights = self.weights_min + np.random.random(
                    (source.neuron_count, target.neuron_count)
                ) * (self.weight_max - self.weight_min)
        else:
            if self.weight_min != -np.inf or self.weight_max != np.inf:
                self.weights = np.clip(
                    self.weights, self.weight_min, self.weight_max
                )

    def calculate(self, spikes: np.array) -> np.array:
        # print("shape", spikes.shape)
        weights_calc = (
            spikes.astype(float).flatten() @ self.weights + self.bias
        )
        # print("calc", weights_calc)
        # print("reshape", weights_calc.reshape(*self.target.shape))
        return weights_calc.reshape(*self.target.shape)

    def update(self, **kwargs) -> NoReturn:
        super().update(**kwargs)

    def reset(self) -> NoReturn:
        super().reset()
