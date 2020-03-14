from src.activation import Activation
import numpy as np


class ReLUActivation(Activation):

    def __init__(self):
        self.last_x: np.ndarray = None

    def forward(self,
                x: np.ndarray):
        self.last_x = x
        res = x.copy()
        res[res < 0] = 0
        return res

    def backward(self,
                 prev_error: np.ndarray):
        res = prev_error.copy()
        res[self.last_x < 0] = 0
        return res

    def __dir__(self):
        return ['last_x']
