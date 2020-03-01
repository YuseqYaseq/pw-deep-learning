import numpy as np

from mlp import MLP
from linear_activaion import LinearActivation


def main():
    batch_size: int = 3
    input_size: int = 5
    output_size: int = 7
    layer = MLP(size_in=input_size,
                size_out=output_size,
                activation=LinearActivation(),
                bias=True)

    input = np.random.rand(batch_size, input_size)
    output = layer.forward(input)
    layer.backward()
    print(output)


if __name__ == '__main__':
    main()
