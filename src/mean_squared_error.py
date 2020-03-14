from src.error import Error
import numpy as np


class MeanSquaredError(Error):

    def __init__(self):
        self.out: np.ndarray = None
        self.y: np.ndarray = None

    def get_error(self,
                  out: np.ndarray,
                  y: np.ndarray):
        self.out = out
        self.y = y
        return np.average((out - y) * (out - y))

    def get_derivative(self):
        error_derivative = 2 * (self.out - self.y)
        return error_derivative
    
    def __repr__(self):
        return '<MeanSquaredError>'