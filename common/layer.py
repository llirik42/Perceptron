import numpy as np

from .activation import Activation

__all__ = ["Layer"]


class Layer:
    __vectorized_activation: np.vectorize
    __vectorized_activation_gradient: np.vectorize
    __weights: np.ndarray
    __bias: np.ndarray
    __x: np.ndarray
    __t: np.ndarray

    def __init__(self, input_size: int, output_size: int, activation: Activation):
        self.__vectorized_activation = np.vectorize(activation)
        self.__vectorized_activation_gradient = np.vectorize(activation.derivative)
        self.__weights = np.zeros((input_size, output_size))
        self.__bias = np.zeros(output_size)

    def forward(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Returns h = F(xW + b) = F(t).

        :param input_vector: x
        :return: h
        """

        self.__x = input_vector.reshape(1, -1)
        result: np.ndarray = np.dot(input_vector, self.__weights) + self.__bias
        self.__t = result.reshape(1, -1)

        return self.__vectorized_activation(result)

    def backward(self, de_dh: np.ndarray, learning_date: float) -> np.ndarray:
        de_dt: np.ndarray = de_dh * self.__vectorized_activation_gradient(self.__t)

        de_dx: np.ndarray = np.dot(de_dt, self.__weights.T)
        de_db: np.ndarray = de_dt[0]  # Convert matrix to vector
        de_dw: np.ndarray = np.dot(self.__x.T, de_dt).reshape(self.__weights.shape)

        self.__weights -= learning_date * de_dw
        self.__bias -= learning_date * de_db

        return de_dx

    def __str__(self) -> str:
        return f"w\n{self.__weights}\n\nb\n{self.__bias}"
