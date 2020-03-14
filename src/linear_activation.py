from src.activation import Activation
import numpy as np


class LinearActivation(Activation):

    def forward(self,
                x: np.ndarray):
        return x

    def backward(self,
                 prev_error: np.ndarray):
        return prev_error

    def __dir__(self):
        return []
