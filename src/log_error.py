from src.error import Error
import numpy as np


class LogError(Error):

    def __init__(self):
        self.out: np.ndarray = None
        self.y: np.ndarray = None

    def get_error(self,
                  out: np.ndarray,
                  y: np.ndarray):
        self.out = out
        self.y = y
        return np.average(-y*np.log(out) - (1 - y) * np.log(1 - out))

    def get_derivative(self):
        error_derivative = -(self.y - self.out) / ((1 - self.out) * self.out)
        return error_derivative
