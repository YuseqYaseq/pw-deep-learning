from src.error import Error
import numpy as np


class MeanAbsoluteError(Error):

    def __init__(self):
        self.out: np.ndarray = None
        self.y: np.ndarray = None

    def get_error(self,
                  out: np.ndarray,
                  y: np.ndarray):
        self.out = out
        self.y = y
        return np.average(abs(out - y))

    def get_derivative(self):
        error_derivative = (self.out > self.y).astype(np.float32)
        error_derivative[error_derivative == 0] = -1
        return error_derivative

    def __dir__(self):
        return ['out', 'y']
