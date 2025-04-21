import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTMCell

from src.models.lstm_model import lstm_cell_step
from src.utils.utils import softmax  # Your from-scratch version


class TestCompareLSTMWithKeras(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        tf.random.set_seed(1)

        self.n_x = 3
        self.n_a = 5
        self.n_y = 2
        self.m = 10

        # Create input & previous states
        self.xt = np.random.randn(self.n_x, self.m)
        self.a_prev = np.random.randn(self.n_a, self.m)
        self.c_prev = np.random.randn(self.n_a, self.m)

        # Create parameters like your from-scratch version
        def rand(shape):
            return np.random.randn(*shape)

        self.parameters = {
            "Wf": rand((self.n_a, self.n_a + self.n_x)),
            "bf": rand((self.n_a, 1)),
            "Wi": rand((self.n_a, self.n_a + self.n_x)),
            "bi": rand((self.n_a, 1)),
            "Wc": rand((self.n_a, self.n_a + self.n_x)),
            "bc": rand((self.n_a, 1)),
            "Wo": rand((self.n_a, self.n_a + self.n_x)),
            "bo": rand((self.n_a, 1)),
            "Wy": rand((self.n_y, self.n_a)),
            "by": rand((self.n_y, 1)),
        }

    def test_compare_with_keras(self):
        # Run your from-scratch LSTM cell
        a_next_np, c_next_np, logits, _ = lstm_cell_step(
            self.xt, self.a_prev, self.c_prev, self.parameters
        )

        yt_pred = softmax(logits)

        # Split weights for Keras
        def split_gate(W):
            return W[:, : self.n_a], W[:, self.n_a :]  # (recurrent, input)

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

        # Build Keras LSTMCell and Dense layer
        cell = LSTMCell(self.n_a)
        dense = Dense(self.n_y)

        x_tf = tf.convert_to_tensor(self.xt.T, dtype=tf.float32)  # (batch, input)
        a_prev_tf = tf.convert_to_tensor(self.a_prev.T, dtype=tf.float32)
        c_prev_tf = tf.convert_to_tensor(self.c_prev.T, dtype=tf.float32)

        cell.build(x_tf.shape)
        cell.set_weights([kernel, recurrent_kernel, bias])

        dense.build((None, self.n_a))
        dense.set_weights([self.parameters["Wy"].T, self.parameters["by"].flatten()])

        # Run Keras LSTM cell
        a_next_tf, [_, c_next_tf] = cell(x_tf, states=[a_prev_tf, c_prev_tf])
        yt_pred_tf = tf.nn.softmax(dense(a_next_tf))

        # Convert to numpy
        a_next_keras = a_next_tf.numpy().T
        c_next_keras = c_next_tf.numpy().T
        yt_pred_keras = yt_pred_tf.numpy().T

        # Compare outputs
        np.testing.assert_allclose(
            a_next_np, a_next_keras, atol=1e-5, err_msg="Mismatch in a_next"
        )
        np.testing.assert_allclose(
            c_next_np, c_next_keras, atol=1e-5, err_msg="Mismatch in c_next"
        )
        np.testing.assert_allclose(
            yt_pred, yt_pred_keras, atol=1e-5, err_msg="Mismatch in yt_pred"
        )

        print("✅ Passed: From-scratch matches Keras LSTMCell")

    def test_compare_with_keras_model_with_softmax(self):
        # Run your from-scratch LSTM cell
        a_next_np, c_next_np, logits, _ = lstm_cell_step(
            self.xt, self.a_prev, self.c_prev, self.parameters
        )

        yt_pred = softmax(logits)

        # Split weights
        def split_gate(W):
            return W[:, : self.n_a], W[:, self.n_a :]  # (recurrent, input)

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

        # Build full Keras RNN + Dense + Softmax model
        cell = LSTMCell(self.n_a)
        rnn = tf.keras.layers.RNN(cell, return_state=True)

        dense_softmax = tf.keras.Sequential(
            [
                Dense(self.n_y, input_shape=(self.n_a,)),
                tf.keras.layers.Activation("softmax"),
            ]
        )

        # Build model components
        x_tf = tf.convert_to_tensor(self.xt.T[:, np.newaxis, :], dtype=tf.float32)
        a_prev_tf = tf.convert_to_tensor(self.a_prev.T, dtype=tf.float32)
        c_prev_tf = tf.convert_to_tensor(self.c_prev.T, dtype=tf.float32)

        cell.build(x_tf.shape)
        cell.set_weights([kernel, recurrent_kernel, bias])

        dense_softmax.build((None, self.n_a))
        dense_softmax.set_weights(
            [self.parameters["Wy"].T, self.parameters["by"].flatten()]
        )

        # Run Keras model with softmax inside the pipeline
        a_next_tf, _, c_next_tf = rnn(x_tf, initial_state=[a_prev_tf, c_prev_tf])
        yt_pred_tf = dense_softmax(a_next_tf)

        # Convert to NumPy for comparison
        a_next_keras = a_next_tf.numpy().T
        c_next_keras = c_next_tf.numpy().T
        yt_pred_keras = yt_pred_tf.numpy().T

        # Compare outputs
        np.testing.assert_allclose(
            a_next_np,
            a_next_keras,
            atol=1e-5,
            err_msg="Mismatch in a_next (model+softmax)",
        )
        np.testing.assert_allclose(
            c_next_np,
            c_next_keras,
            atol=1e-5,
            err_msg="Mismatch in c_next (model+softmax)",
        )
        np.testing.assert_allclose(
            yt_pred,
            yt_pred_keras,
            atol=1e-5,
            err_msg="Mismatch in yt_pred (model+softmax)",
        )

        print("✅ Passed: From-scratch matches Keras model with softmax inside")

    def test_compare_logits_only(self):
        # Run your from-scratch LSTM cell
        a_next_np, _, logits_scratch, _ = lstm_cell_step(
            self.xt, self.a_prev, self.c_prev, self.parameters
        )

        # Prepare weights for Keras
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

        # Build LSTMCell + Dense (no softmax)
        cell = LSTMCell(self.n_a)
        dense = Dense(self.n_y)  # No activation

        x_tf = tf.convert_to_tensor(self.xt.T, dtype=tf.float32)
        a_prev_tf = tf.convert_to_tensor(self.a_prev.T, dtype=tf.float32)
        c_prev_tf = tf.convert_to_tensor(self.c_prev.T, dtype=tf.float32)

        cell.build(x_tf.shape)
        cell.set_weights([kernel, recurrent_kernel, bias])

        dense.build((None, self.n_a))
        dense.set_weights([self.parameters["Wy"].T, self.parameters["by"].flatten()])

        # Forward pass in Keras
        a_next_tf, _ = cell(x_tf, states=[a_prev_tf, c_prev_tf])
        logits_keras = dense(a_next_tf).numpy().T  # shape (n_y, m)

        # Compare logits directly
        np.testing.assert_allclose(
            logits_scratch,
            logits_keras,
            atol=1e-5,
            err_msg="Mismatch in raw logits between scratch and Keras",
        )

        print("✅ Passed: Scratch logits match Keras Dense output (no softmax)")


if __name__ == "__main__":
    unittest.main()
