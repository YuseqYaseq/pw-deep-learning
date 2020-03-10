from src.activation import Activation
import numpy as np


class SoftmaxActivation(Activation):

    def __init__(self):
        self.last_res = None

    def forward(self,
                x: np.ndarray):
        sum = np.sum(np.exp(x), axis=1)
        sum = sum.reshape(sum.size, -1)
        self.last_res = np.exp(x) / sum
        return self.last_res

    def backward(self,
                 prev_error: np.ndarray):
        ret = np.zeros(self.last_res.shape)
        for ex in range(self.last_res.shape[0]):
            gradient = -np.dot(self.last_res[ex, :, None], self.last_res[ex, :, None].T)
            for no in range(gradient.shape[0]):
                gradient[no, no] += self.last_res[ex, no]
            ret[ex] = np.sum(gradient * prev_error[ex], axis=1)[None, :]
        return ret
