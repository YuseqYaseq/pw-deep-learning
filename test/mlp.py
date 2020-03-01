from keras import Sequential
from keras.layers import Dense, K
import numpy as np

from src.sigmoid_activation import SigmoidActivation
from src.linear_activation import LinearActivation
from src.mlp import MLP


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


def test_mlp():
    seed = 1
    eps_decimal = 4
    epsilon = 10 ** (-eps_decimal)

    batch_sizes = [1, 3, 5]
    use_bias = [False, True]
    input_sizes = [1, 3, 5]
    output_sizes = [1, 5, 7]
    activations = [("sigmoid", SigmoidActivation()), ("linear", LinearActivation())]
    for batch_size in batch_sizes:
        for bias in use_bias:
            for input_size in input_sizes:
                for output_size in output_sizes:
                    for activation in activations:

                        model = Sequential()
                        model.add(Dense(output_size,
                                        use_bias=bias,
                                        input_shape=(input_size,),
                                        activation=activation[0]))
                        model.compile(optimizer="sgd", loss="mean_absolute_error")

                        np.random.seed(seed)
                        mlp = MLP(size_in=input_size,
                                  size_out=output_size,
                                  activation=activation[1],
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
                        loss2 = np.average(abs(output2 - y))

                        # equal outputs
                        np.testing.assert_almost_equal(output, output2, decimal=eps_decimal)

                        # equal loss
                        assert(abs(loss - loss2) < epsilon)

                        error_derivative = (output2 > y).astype(np.float32)
                        error_derivative[error_derivative == 0] = -1
                        input_derivative = mlp.backward(error_derivative)

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
