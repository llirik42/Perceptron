from math import exp

from .activation import Activation


class Sigmoid(Activation):
    def __call__(self, x: float) -> float:
        return 1 / (1 + exp(-x))

    def derivative(self, x: float) -> float:
        inverse_exp: float = exp(-x)
        return inverse_exp / ((1 + inverse_exp) ** 2)
