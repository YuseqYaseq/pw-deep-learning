from src.relu_activation import ReLUActivation
from src.mean_absolute_error import MeanAbsoluteError
from test.common import compared_to_keras


def test_mean_absolute_error():
    batch_sizes = [1, 3, 5]
    input_sizes = [1, 3, 5]
    output_sizes = [1, 3, 5]
    use_bias = [False, True]
    for batch_size in batch_sizes:
        for bias in use_bias:
            for input_size in input_sizes:
                for output_size in output_sizes:
                    compared_to_keras(batch_size, input_size, output_size, ReLUActivation(),
                                      MeanAbsoluteError(), bias)
