import numpy as np
import pandas as pd

from common import Normalizer, Layer


class TemperatureModel:
    __layers: list[Layer]
    __normalizer: Normalizer
    __target: str
    __train_errors: list[float]

    def __init__(self, layers: list[Layer], df: pd.DataFrame, target: str):
        self.__layers = layers
        self.__normalizer = Normalizer(df)
        self.__target = target
        self.__train_errors = []

    def push_train_error(self, error: float) -> None:
        self.__train_errors.append(error)

    @property
    def train_errors(self) -> list[float]:
        return self.__train_errors

    def forward(self, inputs: pd.Series) -> float:
        result: np.ndarray = np.array(self.__normalizer.normalize(inputs))
        for layer in self.__layers:
            result: np.ndarray = layer.forward(result)
        return self.__normalizer.denormalize_feature(value=result[0], feature_name=self.__target)

    def backward(self, error: float, learning_rate: float) -> None:
        current_output: np.ndarray = np.array([error])
        for layer in self.__layers[::-1]:
            current_output: np.ndarray = layer.backward(current_output, learning_rate)
