from keras import Sequential
from keras.layers import Dense
import numpy as np

from src.log_error import LogError
from src.network import Network
from src.sigmoid_activation import SigmoidActivation
from test.common import get_weight_grad


def test_networks():

    input_size = 20
    output_size = 3
    batch_size = 5
    eps_decimal = 4
    seed = 1
    layers = [(10, SigmoidActivation(), True),
              (output_size, SigmoidActivation(), True)]
    network = Network(input_size, layers, LogError(), seed)

    model = Sequential()
    model.add(Dense(10,
                    use_bias=True,
                    input_shape=(input_size,),
                    activation="sigmoid"))
    model.add(Dense(output_size,
                    use_bias=True,
                    input_shape=(input_size,),
                    activation="sigmoid"))
    model.compile(optimizer="sgd", loss="binary_crossentropy")

    model.layers[0].set_weights([network.layers[0].w, network.layers[0].b.flatten()])
    model.layers[1].set_weights([network.layers[1].w, network.layers[1].b.flatten()])

    x = np.random.rand(batch_size, input_size)
    y = np.random.rand(batch_size, output_size)

    loss = model.evaluate(x, y, verbose=2)

    output = model.predict(x)
    output2 = network.predict(x)
    loss2 = network.evaluate(x, y)

    # equal outputs
    np.testing.assert_almost_equal(output, output2, decimal=eps_decimal)

    # equal loss
    np.testing.assert_almost_equal(loss, loss2, decimal=eps_decimal)

    # equal weights and biases derivatives
    [dw0, db0, dw1, db1] = get_weight_grad(model, x, y)
    np.testing.assert_almost_equal(db1, network.layers[1].db, decimal=eps_decimal)
    np.testing.assert_almost_equal(dw1, network.layers[1].dw, decimal=eps_decimal)
    np.testing.assert_almost_equal(db0, network.layers[0].db, decimal=eps_decimal)
    np.testing.assert_almost_equal(dw0, network.layers[0].dw, decimal=eps_decimal)
