import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input, SimpleRNN
from tensorflow.keras.models import Model

from src.rnn_model import initialize_rnn_parameters, rnn_forward
from src.utils import softmax


class TestRNNForwardVsKeras(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        tf.random.set_seed(3)

        self.n_a = 5
        self.vocab_size = 3
        self.n_y = self.vocab_size
        self.T_x = 6

        self.X = np.random.randint(0, self.vocab_size, size=self.T_x).tolist()
        self.a0 = np.random.randn(self.n_a, 1).astype(np.float32)

        self.parameters = initialize_rnn_parameters(
            self.n_a, self.vocab_size, self.n_y, seed=3
        )

    def keras_rnn_with_dense(self):
        """Build Keras RNN + Dense model with scratch weights."""
        model = tf.keras.Sequential(
            [
                SimpleRNN(
                    self.n_a,
                    activation="tanh",
                    return_sequences=True,
                    input_shape=(self.T_x, self.vocab_size),
                ),
                Dense(self.n_y),
            ]
        )
        model.build()

        # Set weights from scratch model
        model.layers[0].set_weights(
            [
                self.parameters["Wax"].T,
                self.parameters["Waa"].T,
                self.parameters["ba"].flatten(),
            ]
        )
        model.layers[1].set_weights(
            [
                self.parameters["Wya"].T,
                self.parameters["by"].flatten(),
            ]
        )

        return model

    def test_forward_outputs_match_softmax(self):
        """Compare softmax predictions from scratch vs. Keras model."""
        _, _, logits, _ = rnn_forward(self.X, self.a0, self.parameters)
        y_hat_np = softmax(logits)  # shape (n_y, T_x)

        # Prepare Keras input (batch-first)
        x_input = np.expand_dims(np.eye(self.vocab_size)[self.X], axis=0).astype(
            np.float32
        )

        # Keras functional model with explicit softmax
        inputs = Input(shape=(self.T_x, self.vocab_size))
        rnn = SimpleRNN(self.n_a, return_sequences=True, return_state=True)
        dense = Dense(self.vocab_size)
        softmax_activation = Activation("softmax")

        rnn_out, _ = rnn(inputs)
        logits = dense(rnn_out)
        y_pred = softmax_activation(logits)
        model = Model(inputs=inputs, outputs=y_pred)

        rnn.set_weights(
            [
                self.parameters["Wax"].T,
                self.parameters["Waa"].T,
                self.parameters["ba"].flatten(),
            ]
        )
        dense.set_weights(
            [
                self.parameters["Wya"].T,
                self.parameters["by"].flatten(),
            ]
        )

        keras_output = model.predict(x_input, verbose=0)[0].T  # (n_y, T_x)

        print("\n🧪 Forward output comparison (softmax predictions):")
        print("From-scratch:\n", np.round(y_hat_np, 4))
        print("Keras:\n", np.round(keras_output, 4))
        print("Max diff:", np.abs(y_hat_np - keras_output).max())

        np.testing.assert_allclose(
            y_hat_np, keras_output, atol=1.5e-3, err_msg="Mismatch in y_hat predictions"
        )
        print("\n✅ Passed: rnn_forward matches Keras softmax predictions")

    def test_forward_logits_match(self):
        """Compare raw output logits from scratch vs. Keras."""
        _, _, logits, _ = rnn_forward(self.X, self.a0, self.parameters)

        model = self.keras_rnn_with_dense()

        x_input = np.expand_dims(np.eye(self.vocab_size)[self.X], axis=0).astype(
            np.float32
        )
        x_tf = tf.convert_to_tensor(x_input)

        logits_tf = model(x_tf, training=False)  # shape: (1, T_x, n_y)
        z_y_keras = tf.transpose(logits_tf[0], perm=[1, 0]).numpy()  # (n_y, T_x)

        print("\n📊 Logits comparison:")
        print("From-scratch:\n", np.round(logits, 4))
        print("Keras:\n", np.round(z_y_keras, 4))
        print("Max diff:", np.abs(logits - z_y_keras).max())

        np.testing.assert_allclose(
            logits, z_y_keras, atol=1.5e-3, err_msg="Mismatch in logits (z_y)"
        )
        print("\n✅ Passed: From-scratch logits match Keras logits")


if __name__ == "__main__":
    unittest.main()
