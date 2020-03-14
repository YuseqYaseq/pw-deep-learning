from abc import ABC, abstractmethod
import numpy as np


class Error(ABC):

    @abstractmethod
    def get_error(self,
                  out: np.ndarray,
                  y: np.ndarray):
        pass

    @abstractmethod
    def get_derivative(self):
        pass

    @abstractmethod
    def __dir__(self):
        pass
