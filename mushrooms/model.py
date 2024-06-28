import numpy as np
import pandas as pd

from common import Layer


class TemperatureModel:
    __layers: list[Layer]

    def __init__(self, layers: list[Layer]):
        self.__layers = layers

    def forward(self, inputs: pd.Series) -> float:
        result: np.ndarray = np.array(inputs)
        for layer in self.__layers:
            result: np.ndarray = layer.forward(result)
        return result[0]

    def backward(self, error: float, learning_rate: float) -> None:
        current_output: np.ndarray = np.array([error])
        for layer in self.__layers[::-1]:
            current_output: np.ndarray = layer.backward(current_output, learning_rate)
