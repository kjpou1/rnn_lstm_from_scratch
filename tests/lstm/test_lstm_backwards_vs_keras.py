import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.models import Model

from src.models.lstm_model import (
    initialize_lstm_parameters,
    lstm_backwards,
    lstm_forward,
)
from src.utils.utils import softmax


class TestLSTMBackwardsVsKeras(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        tf.random.set_seed(3)

        self.n_x = 3  # vocab size
        self.n_a = 5
        self.n_y = 2
        self.m = 4
        self.T_x = 6

        # 🧠 Use index-based input instead of fake one-hot
        self.x_np = np.random.randint(
            0, self.n_x, size=(self.T_x,), dtype=np.int32
        )  # (T_x,)

        # 👇 Simulate a single batch (m=1) for Keras (since scratch is not batched)
        one_hot = np.zeros((self.T_x, self.n_x), dtype=np.float32)
        one_hot[np.arange(self.T_x), self.x_np] = 1.0
        self.x_tf = tf.convert_to_tensor(
            one_hot[np.newaxis, :, :]
        )  # shape (1, T_x, n_x)

        self.a0_np = np.random.randn(self.n_a, 1).astype(np.float32)
        self.a0_tf = tf.convert_to_tensor(self.a0_np.T)  # shape (1, n_a)

        # From-scratch params
        self.parameters = initialize_lstm_parameters(
            self.n_a, self.n_x, self.n_y, seed=3
        )

        # Forward pass (your LSTM expects index-based input now)
        self.a_out, self.y_out, self.caches = lstm_forward(
            self.x_np, self.a0_np, self.parameters
        )

        # Upstream gradient w.r.t. all hidden states
        self.da = np.random.randn(self.n_a, 1, self.T_x).astype(np.float32)

    def keras_lstm_setup(self):
        inputs = Input(shape=(self.T_x, self.n_x))
        lstm = LSTM(self.n_a, return_sequences=True, return_state=True)
        outputs, h, c = lstm(inputs)
        model = Model(inputs=inputs, outputs=[outputs, h, c])

        # Split weights from our model
        def split_gate(W):
            return W[:, : self.n_a], W[:, self.n_a :]

        Wi_rec, Wi_in = split_gate(self.parameters["Wi"])
        Wf_rec, Wf_in = split_gate(self.parameters["Wf"])
        Wc_rec, Wc_in = split_gate(self.parameters["Wc"])
        Wo_rec, Wo_in = split_gate(self.parameters["Wo"])

        kernel = np.concatenate([Wi_in.T, Wf_in.T, Wc_in.T, Wo_in.T], axis=1)
        recurrent_kernel = np.concatenate(
            [Wi_rec.T, Wf_rec.T, Wc_rec.T, Wo_rec.T], axis=1
        )
        bias = np.concatenate(
            [
                self.parameters["bi"].flatten(),
                self.parameters["bf"].flatten(),
                self.parameters["bc"].flatten(),
                self.parameters["bo"].flatten(),
            ]
        )

        lstm.build(self.x_tf.shape)
        lstm.set_weights([kernel, recurrent_kernel, bias])

        return model, lstm

    def test_compare_gradients_to_keras(self):
        model, lstm_layer = self.keras_lstm_setup()

        with tf.GradientTape() as tape:
            tape.watch([self.x_tf, self.a0_tf])

            output, h, c = lstm_layer(
                self.x_tf, initial_state=[self.a0_tf, tf.zeros_like(self.a0_tf)]
            )

            da_tf = tf.convert_to_tensor(
                self.da.transpose(1, 2, 0)
            )  # shape (m, T_x, n_a)
            loss = tf.reduce_sum(output * da_tf)

        grads = tape.gradient(
            loss, lstm_layer.trainable_variables + [self.x_tf, self.a0_tf]
        )
        keras_grads = grads[:3]  # kernel, recurrent_kernel, bias
        keras_dx = grads[3].numpy().transpose(2, 0, 1)  # back to (n_x, m, T_x)
        keras_da0 = grads[4].numpy().T  # (n_a, m)

        # Get your gradients
        my_grads = lstm_backwards(self.da, self.caches)

        # Extract Keras gate-wise gradients
        keras_dkernel = keras_grads[0].numpy()
        keras_drecurrent = keras_grads[1].numpy()
        keras_dbias = keras_grads[2].numpy()

        def unsplit_gates(matrix):
            return np.split(matrix, 4, axis=1)

        dWi_in_T, dWf_in_T, dWc_in_T, dWo_in_T = unsplit_gates(keras_dkernel)
        dWi_rec_T, dWf_rec_T, dWc_rec_T, dWo_rec_T = unsplit_gates(keras_drecurrent)

        dWi = np.concatenate([dWi_rec_T.T, dWi_in_T.T], axis=1)
        dWf = np.concatenate([dWf_rec_T.T, dWf_in_T.T], axis=1)
        dWc = np.concatenate([dWc_rec_T.T, dWc_in_T.T], axis=1)
        dWo = np.concatenate([dWo_rec_T.T, dWo_in_T.T], axis=1)

        dbi, dbf, dbc, dbo = np.split(keras_dbias, 4)

        # Compare gate weights & biases
        np.testing.assert_allclose(
            my_grads["dWi"], dWi, atol=1e-5, err_msg="Mismatch in dWi"
        )
        np.testing.assert_allclose(
            my_grads["dWf"], dWf, atol=1e-5, err_msg="Mismatch in dWf"
        )
        np.testing.assert_allclose(
            my_grads["dWc"], dWc, atol=1e-5, err_msg="Mismatch in dWc"
        )
        np.testing.assert_allclose(
            my_grads["dWo"], dWo, atol=1e-5, err_msg="Mismatch in dWo"
        )

        np.testing.assert_allclose(
            my_grads["dbi"].flatten(), dbi, atol=1e-5, err_msg="Mismatch in dbi"
        )
        np.testing.assert_allclose(
            my_grads["dbf"].flatten(), dbf, atol=1e-5, err_msg="Mismatch in dbf"
        )
        np.testing.assert_allclose(
            my_grads["dbc"].flatten(), dbc, atol=1e-5, err_msg="Mismatch in dbc"
        )
        np.testing.assert_allclose(
            my_grads["dbo"].flatten(), dbo, atol=1e-5, err_msg="Mismatch in dbo"
        )

        # Compare dx and da0
        np.testing.assert_allclose(
            my_grads["da0"], keras_da0, atol=1e-5, err_msg="Mismatch in da0"
        )

        print(
            "✅ Passed: Custom lstm_backwards gradients match Keras (weights + inputs + initial state)"
        )


if __name__ == "__main__":
    unittest.main()
