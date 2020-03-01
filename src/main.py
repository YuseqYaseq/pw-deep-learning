import numpy as np

from src.mean_absolute_error import MeanAbsoluteError
from src.network import Network
from src.linear_activation import LinearActivation
from src.sigmoid_activation import SigmoidActivation


def main():
    batch_size: int = 10
    input_size: int = 20
    output_size: int = 3
    alpha: float = 0.01
    layers = [(50, SigmoidActivation(), True),
              (10, SigmoidActivation(), True),
              (3, LinearActivation(), True)]

    error = MeanAbsoluteError()
    network = Network(input_size, layers, error)

    x = np.random.rand(batch_size, input_size)
    y = np.random.rand(batch_size, output_size)
    for i in range(1000):
        loss = network.fit(x, y, alpha)
        print("{0} iteration, loss = {1}.".format(i, loss))


if __name__ == '__main__':
    main()
