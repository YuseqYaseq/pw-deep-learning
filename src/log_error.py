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
        with np.errstate(divide='ignore', invalid='ignore'):
            calc = -y*np.log(out) - (1.0 - y) * np.log(1 - out)
            calc[np.isnan(calc)] = 0.0
            return np.average(calc)

    def get_derivative(self):

        with np.errstate(divide='ignore', invalid='ignore'):
            error_derivative = -(self.y - self.out) / ((1.0 - self.out) * self.out)
            error_derivative[np.isnan(error_derivative)] = 0.0
            return error_derivative

    def __repr__(self):
        return '<LogError>'