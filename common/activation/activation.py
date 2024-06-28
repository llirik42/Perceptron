from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: float) -> float:
        pass

    @abstractmethod
    def derivative(self, x: float) -> float:
        pass
