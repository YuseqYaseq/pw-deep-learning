from src.activation import Activation
import numpy as np


class SigmoidActivation(Activation):

    def __init__(self):
        self.last_x = None

    def forward(self,
                x: np.ndarray):
        self.last_x = x
        return 1 / (1 + np.exp(-x))

    def backward(self,
                 prev_error: np.ndarray):
        emx = np.exp(-self.last_x)
        return (emx / ((emx + 1) * (emx + 1))) * prev_error

    def __dir__(self):
        return ['last_x']
        
    def __repr__(self):
        return '<SigmoidActivation>'
