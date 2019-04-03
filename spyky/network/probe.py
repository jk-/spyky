import numpy as np
from abc import ABC, abstractmethod
from typing import NoReturn, Union, Iterable, Any, Optional
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
    ) -> NoReturn:
        super().__init__()

        self.target = target
        self.variables = variables
        self.reset()

    def get(self, item: str) -> Any:
        return np.array(self.data[item])

    def save(self) -> NoReturn:
        for v in self.variables:
            self.data[v].append(self.target.__dict__[v].flatten())

    def reset(self) -> NoReturn:
        self.data = {v: [] for v in self.variables}
