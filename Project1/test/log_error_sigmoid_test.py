import numpy as np

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


def test_log_for_perfect_predictions():
    out = np.array([1, 0])
    y = np.array([1, 0])

    error = LogError()
    loss = error.get_error(out, y)
    derivative = error.get_derivative()
    np.testing.assert_almost_equal(loss, 0.0)
    np.testing.assert_almost_equal(derivative, 0.0)


def test_log_for_opposite_predictions():
    out = np.array([1, 0])
    y = np.array([0, 1])

    error = LogError()
    loss = error.get_error(out, y)
    derivative = error.get_derivative()
    np.testing.assert_almost_equal(loss, np.inf)
    np.testing.assert_almost_equal(derivative, np.array([np.inf, -np.inf]))
