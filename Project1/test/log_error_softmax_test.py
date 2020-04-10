from src.log_error import LogError
from src.softmax_activation import SoftmaxActivation
from test.common import compared_to_keras


def test_softmax():
    batch_sizes = [1, 4]
    input_sizes = [3]
    output_sizes = [3, 5]
    use_bias = [False, True]
    for batch_size in batch_sizes:
        for bias in use_bias:
            for input_size in input_sizes:
                for output_size in output_sizes:
                    compared_to_keras(batch_size, input_size, output_size, SoftmaxActivation(),
                                      LogError(), bias)
