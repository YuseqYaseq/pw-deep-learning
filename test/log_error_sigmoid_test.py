from src.log_error import LogError
from src.sigmoid_activation import SigmoidActivation
from test.common import compared_to_keras


def test_log_error():
    batch_sizes = [1, 3, 5]
    input_sizes = [1, 3, 5]
    output_sizes = [1, 3, 5]
    use_bias = [False, True]
    for batch_size in batch_sizes:
        for bias in use_bias:
            for input_size in input_sizes:
                for output_size in output_sizes:
                    compared_to_keras(batch_size, input_size, output_size, SigmoidActivation(),
                                      LogError(), bias)
