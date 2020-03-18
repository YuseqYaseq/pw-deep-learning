import numpy as np

from src.activation import Activation
from src.layer import Layer


class MLP(Layer):
    def __init__(self,
                 size_in: int,
                 size_out: int,
                 activation: Activation,
                 bias: bool):
        self.w = np.random.randn(size_in, size_out) * np.sqrt(2 / (size_in + size_out))
        self.w_momentum = 0
        if bias:
            self.b = np.random.randn(1, size_out) * np.sqrt(1 / size_out)
            self.b_momentum = 0
        else:
            self.b = None

        self.a = activation

        self.last_x = None
        self.activated_w = None
        self.dw = None
        self.db = None

    def forward(self,
                x: np.ndarray):
        self.last_x = x
        if self.b is not None:
            self.activated_w = self.a(np.dot(x, self.w) + self.b)
        else:
            self.activated_w = self.a(np.dot(x, self.w))
        return self.activated_w

    def backward(self,
                 prev_error: np.ndarray):
        prev_error = self.a.backward(prev_error)
        self.dw = np.dot(self.last_x.T, prev_error) / (self.last_x.shape[0] * self.w.shape[1])
        if self.b is not None:
            self.db = np.average(prev_error / self.b.shape[1], axis=0)
        return np.dot(prev_error, self.w.T) * self.w.shape[0] / self.w.shape[1]

    def update_parameters(self,
                          alpha: float,
                          beta: float):
        self.w_momentum = beta * self.w_momentum - alpha * self.dw
        self.w += self.w_momentum
        if self.b is not None:
            self.b_momentum = beta * self.b_momentum - alpha * self.db
            self.b += self.b_momentum

    def __repr__(self):
        return 'Layer({}, bias={}, {})'.format(self.w.shape, False if self.b is None else True, self.a)