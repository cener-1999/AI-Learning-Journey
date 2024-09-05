import numpy as np
from typing import List, Callable
from dataclasses import dataclass


class ActivationFunction(object):
    @staticmethod
    def step_function(x: np.ndarray):
        y = x > 0
        return y.astype(np.int64)

    @staticmethod
    def sigmoid_function(x: np.ndarray):
        y = 1 / (np.exp(-x) + 1)
        return y

    @staticmethod
    def relu_function(x: np.ndarray):
        return np.maximum(x, 0)


class SigmaFunction(object):
    @staticmethod
    def identity_function(x: np.ndarray):
        return x


@dataclass
class NetworkLayer:
    W: np.array
    B: np.array
    func: Callable

    def __init__(self, W, B, func):
        self.W = W
        self.B = B
        self.func = func


class NeuralNetwork:
    def __init__(self, layers: List[NetworkLayer]):
        self.layers = layers

    def front_propagation(self, input_data: np.ndarray) -> np.array:
        X = input_data
        for layer in self.layers:
            X = layer.func(np.dot(X, layer.W) + layer.B)
        return X

    def back_propagation(self):
        pass


if __name__ == '__main__':
    W1 = np.array([[0.1, 0.3, 0.5],
                   [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    a1 = ActivationFunction.sigmoid_function

    W2 = np.array([[0.1, 0.4],
                   [0.2, 0.5],
                   [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    a2 = ActivationFunction.sigmoid_function

    W3 = np.array([[0.1, 0.3],
                   [0.2, 0.4]])
    b3 = np.array([0.1, 0.2])
    sigma = SigmaFunction.identity_function

    l1 = NetworkLayer(W1, b1, a1)
    l2 = NetworkLayer(W2, b2, a2)
    l3 = NetworkLayer(W3, b3, sigma)

    network = NeuralNetwork([l1, l2, l3])

    input_data = np.array([1.0, 0.5])
    res = network.front_propagation(input_data)
    print(res)
