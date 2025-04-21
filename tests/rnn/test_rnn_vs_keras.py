import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNNCell

from src.models.rnn_model import rnn_cell_step
from src.utils.utils import softmax


class TestCompareRNNWithKeras(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        tf.random.set_seed(1)

        self.n_x = 3
        self.n_a = 5
        self.n_y = 2
        self.m = 10

        # Inputs and initial hidden state
        self.xt = np.random.randn(self.n_x, self.m)
        self.a_prev = np.random.randn(self.n_a, self.m)

        def rand(shape):
            return np.random.randn(*shape)

        self.parameters = {
            "Wax": rand((self.n_a, self.n_x)),
            "Waa": rand((self.n_a, self.n_a)),
            "ba": rand((self.n_a, 1)),
            "Wya": rand((self.n_y, self.n_a)),
            "by": rand((self.n_y, 1)),
        }

    def test_compare_with_keras_softmax(self):
        # From-scratch forward
        a_next_np, logits, *_ = rnn_cell_step(self.xt, self.a_prev, self.parameters)
        yt_pred_np = softmax(logits)  # Apply softmax manually

        # Keras forward
        kernel = self.parameters["Wax"].T
        recurrent_kernel = self.parameters["Waa"].T
        bias = self.parameters["ba"].flatten()

        dense = Dense(self.n_y)
        dense.build((None, self.n_a))
        dense.set_weights([self.parameters["Wya"].T, self.parameters["by"].flatten()])

        x_tf = tf.convert_to_tensor(self.xt.T, dtype=tf.float32)
        a_prev_tf = tf.convert_to_tensor(self.a_prev.T, dtype=tf.float32)

        cell = SimpleRNNCell(self.n_a, activation="tanh")
        cell.build(x_tf.shape)
        cell.set_weights([kernel, recurrent_kernel, bias])

        a_next_tf, _ = cell(x_tf, [a_prev_tf])
        yt_pred_tf = tf.nn.softmax(dense(a_next_tf))
        yt_pred_keras = yt_pred_tf.numpy().T

        # Logs
        print("\n🧪 Input shapes:")
        print(f"x_t: {self.xt.shape}, a_prev: {self.a_prev.shape}")

        print("\n🔁 Output comparison (a_next):")
        print("From-scratch:\n", np.round(a_next_np, 4))
        print("Keras:\n", np.round(a_next_tf.numpy().T, 4))
        print("Max diff:", np.abs(a_next_np - a_next_tf.numpy().T).max())

        print("\n📊 Output comparison (y_pred):")
        print("From-scratch:\n", np.round(yt_pred_np, 4))
        print("Keras:\n", np.round(yt_pred_keras, 4))
        print("Max diff:", np.abs(yt_pred_np - yt_pred_keras).max())

        # Assertions
        np.testing.assert_allclose(
            a_next_np, a_next_tf.numpy().T, atol=1e-5, err_msg="Mismatch in a_next"
        )
        np.testing.assert_allclose(
            yt_pred_np, yt_pred_keras, atol=1e-5, err_msg="Mismatch in y_pred"
        )

        print("\n✅ Passed: From-scratch RNNCell matches Keras SimpleRNNCell softmax")

    def test_compare_with_keras_logits(self):
        # From-scratch logits
        a_next_np, logits, *_ = rnn_cell_step(self.xt, self.a_prev, self.parameters)

        # Keras logits
        kernel = self.parameters["Wax"].T
        recurrent_kernel = self.parameters["Waa"].T
        bias = self.parameters["ba"].flatten()

        dense = Dense(self.n_y, activation=None)
        dense.build((None, self.n_a))
        dense.set_weights([self.parameters["Wya"].T, self.parameters["by"].flatten()])

        x_tf = tf.convert_to_tensor(self.xt.T, dtype=tf.float32)
        a_prev_tf = tf.convert_to_tensor(self.a_prev.T, dtype=tf.float32)

        cell = SimpleRNNCell(self.n_a, activation="tanh")
        cell.build(x_tf.shape)
        cell.set_weights([kernel, recurrent_kernel, bias])

        a_next_tf, _ = cell(x_tf, [a_prev_tf])
        z_y_keras = dense(a_next_tf).numpy().T

        print("\n🧪 [LOGITS] Output comparison (z_y):")
        print("From-scratch:\n", np.round(logits, 4))
        print("Keras:\n", np.round(z_y_keras, 4))
        print("Max diff:", np.abs(logits - z_y_keras).max())

        np.testing.assert_allclose(
            logits, z_y_keras, atol=1e-5, err_msg="Mismatch in output logits"
        )

        print("\n✅ Passed: From-scratch logits match Keras Dense output (no softmax)")


if __name__ == "__main__":
    unittest.main()
