import numpy as np
from keras import Sequential
from keras.layers import Dense, K

from src.activation import Activation
from src.error import Error
from src.linear_activation import LinearActivation
from src.log_error import LogError
from src.mean_absolute_error import MeanAbsoluteError
from src.mean_squared_error import MeanSquaredError
from src.mlp import MLP
from src.sigmoid_activation import SigmoidActivation


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def compared_to_keras(batch_size: int,
                      input_size: int,
                      output_size: int,
                      activation: Activation,
                      loss_function: Error,
                      bias: bool,
                      eps_decimal: int = 4,
                      seed: int = 1):

    if isinstance(activation, SigmoidActivation):
        activation_name = "sigmoid"
    elif isinstance(activation, LinearActivation):
        activation_name = "linear"
    else:
        raise RuntimeError("Unknown activation function!")

    if isinstance(loss_function, MeanAbsoluteError):
        loss_function_name = "mean_absolute_error"
    elif isinstance(loss_function, LogError):
        loss_function_name = "binary_crossentropy"
    elif isinstance(loss_function, MeanSquaredError):
        loss_function_name = "mean_squared_error"
    else:
        raise RuntimeError("Unknown loss function!")

    model = Sequential()
    model.add(Dense(output_size,
                    use_bias=bias,
                    input_shape=(input_size,),
                    activation=activation_name))
    model.compile(optimizer="sgd", loss=loss_function_name)

    np.random.seed(seed)
    mlp = MLP(size_in=input_size,
              size_out=output_size,
              activation=activation,
              bias=bias,
              seed=seed)
    if bias:
        model.layers[0].set_weights([mlp.w, mlp.b.flatten()])
    else:
        model.layers[0].set_weights([mlp.w])

    x = np.random.rand(batch_size, input_size)
    y = np.random.rand(batch_size, output_size)

    loss = model.evaluate(x, y, verbose=2)

    output = model.predict(x)
    output2 = mlp.forward(x)
    loss2 = loss_function.get_error(output2, y)

    # equal outputs
    np.testing.assert_almost_equal(output, output2, decimal=eps_decimal)

    # equal loss
    np.testing.assert_almost_equal(loss, loss2, decimal=eps_decimal)

    derivative = loss_function.get_derivative()
    mlp.backward(derivative)

    # equal weights and biases derivatives
    if bias:
        [dw, db] = get_weight_grad(model, x, y)
        np.testing.assert_almost_equal(db, mlp.db, decimal=eps_decimal)
        np.testing.assert_almost_equal(dw, mlp.dw, decimal=eps_decimal)
    else:
        [dw] = get_weight_grad(model, x, y)
        np.testing.assert_almost_equal(dw, mlp.dw, decimal=eps_decimal)

    # equal input derivatives
    dx = get_layer_output_grad(model, x, y)  # TODO
