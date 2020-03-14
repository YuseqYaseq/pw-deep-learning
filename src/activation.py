from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):

    def __call__(self, x):
        return self.forward(x)

    @abstractmethod
    def forward(self,
                x: np.ndarray):
        pass

    @abstractmethod
    def backward(self,
                 prev_error: np.ndarray):
        pass

    @abstractmethod
    def __dir__(self):
        pass
