import numpy as np
from abc import ABC, abstractmethod
from typing import NoReturn, Union, Iterable, Any
from spyky.network import AbstractLayer, AbstractConnection


class AbstractProbe(ABC):
    def __init__(self) -> NoReturn:
        super().__init__()

    @abstractmethod
    def get(self, item: str) -> Any:
        pass

    @abstractmethod
    def save(self) -> NoReturn:
        psas

    @abstractmethod
    def reset(self) -> NoReturn:
        pass


class Probe:
    def __init__(
        self,
        target: Union[AbstractLayer, AbstractConnection],
        variables: Iterable[str],
        length: int,
    ) -> NoReturn:
        super().__init__()

        self.target = target
        self.variables = variables
        self.length = length
        self.reset()

    def get(self, item: str) -> Any:
        return self.data[item]

    def save(self) -> NoReturn:
        for var in self.variables:
            self.data[var] = np.expand_dims(self.target.__dict__[var], 1)

    def reset(self) -> NoReturn:
        self.data = {
            var: np.zeros(
                self.target.__dict__[var].size,
                dtype=self.target.__dict__[var].dtype,
            )
            for var in self.variables
        }
