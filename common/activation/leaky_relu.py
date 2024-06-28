from .activation import Activation


class LeakyReLu(Activation):
    def __call__(self, x: float) -> float:
        return x if x >= 0 else 0.1 * x

    def derivative(self, x: float) -> float:
        return 1 if x >= 0 else 0.1
