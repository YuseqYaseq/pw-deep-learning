from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self,
                x: np.ndarray):
        pass

    @abstractmethod
    def backward(self,
                 prev_error: np.ndarray):
        pass

    @abstractmethod
    def update_parameters(self,
                          alpha: float,
                          beta: float):
        pass
