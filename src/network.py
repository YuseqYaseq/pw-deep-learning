from typing import Tuple, List
import numpy as np

from src.activation import Activation
from src.error import Error
from src.mlp import MLP


class Network:

    def __init__(self,
                 input_size: int,
                 layers: List[Tuple[int, Activation, bool]],
                 error_fun: Error):

        self.layers = []
        for i in range(len(layers)):
            if i == 0:
                self.layers.append(MLP(input_size, layers[i][0], layers[i][1], layers[i][2]))
            else:
                self.layers.append(MLP(layers[i-1][0], layers[i][0], layers[i][1], layers[i][2]))
        self.error_fun = error_fun

    def predict(self,
                x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            alpha: float):
        out = self.predict(x)
        loss = self.error_fun.get_error(out, y)
        derivative = self.error_fun.get_derivative()
        for layer in list(reversed(self.layers)):
            derivative = layer.backward(derivative)
        for layer in self.layers:
            layer.update_parameters(alpha)
        return loss