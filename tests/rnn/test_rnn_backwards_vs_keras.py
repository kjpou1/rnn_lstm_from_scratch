import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input, SimpleRNN
from tensorflow.keras.models import Model

from src.models.rnn_model import initialize_rnn_parameters, rnn_backward, rnn_forward


def print_matrix_diff(mat1, mat2, name="Matrix"):
    print(f"\n🧪 {name} comparison:")

    mat1 = np.atleast_2d(mat1)
    mat2 = np.atleast_2d(mat2)
    diff = mat1 - mat2

    print(f"\n🔸 From-scratch:")
    for row in mat1:
        print(" ".join(f"{val: .6f}" for val in row))

    print(f"\n🔹 Keras:")
    for row in mat2:
        print(" ".join(f"{val: .6f}" for val in row))

    print(f"\n⚠️  Diff (yours - keras):")
    for row in diff:
        print(" ".join(f"{val: .6f}" for val in row))

    print(f"\n📊 Max abs diff: {np.max(np.abs(diff))}")


class TestRNNBackwardsVsKeras(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        tf.random.set_seed(3)

        self.n_a = 5
        self.m = 1  # single example
        self.T_x = 6
        self.vocab_size = 3
        self.n_x = self.vocab_size
        self.n_y = self.vocab_size  # softmax over vocab

        # Input and target sequences (token IDs)
        self.X = np.random.randint(0, self.vocab_size, size=self.T_x).tolist()
        self.Y = np.random.randint(0, self.vocab_size, size=self.T_x).tolist()

        # Initial hidden state
        self.a0 = np.random.randn(self.n_a, self.m).astype(np.float32)

        # Parameters
        self.parameters = initialize_rnn_parameters(
            self.n_a, self.vocab_size, self.n_y, seed=3
        )

        # Forward pass
        self.a_out, self.x_cache, self.logits, self.z_t = rnn_forward(
            self.X, self.a0, self.parameters
        )
        self.cache = (self.a_out, self.x_cache, self.logits, self.z_t)

    def keras_rnn_with_dense(self):
        # Keras model: SimpleRNN → Dense → Softmax
        inputs = Input(shape=(self.T_x, self.n_x))
        rnn = SimpleRNN(self.n_a, return_sequences=True, return_state=True)
        dense = Dense(self.n_y)
        softmax = Activation("softmax")

        x = inputs
        rnn_out, _ = rnn(x)
        logits = dense(rnn_out)
        y_pred = softmax(logits)

        model = Model(inputs=inputs, outputs=y_pred)

        # Set weights from scratch model
        kernel = self.parameters["Wax"].T  # (input_dim, units)
        recurrent_kernel = self.parameters["Waa"].T  # (units, units)
        bias = self.parameters["ba"].flatten()

        dense_W = self.parameters["Wya"].T
        dense_b = self.parameters["by"].flatten()

        rnn.build((self.m, self.T_x, self.n_x))
        rnn.set_weights([kernel, recurrent_kernel, bias])
        dense.set_weights([dense_W, dense_b])

        return model

    def test_compare_gradients_to_keras(self):
        # One-hot encode input
        x_input = np.stack([np.eye(self.vocab_size)[self.X]] * self.m).astype(
            np.float32
        )  # (m, T_x, vocab)
        y_true = np.stack([np.eye(self.vocab_size)[self.Y]] * self.m).astype(
            np.float32
        )  # (m, T_x, vocab)

        model = self.keras_rnn_with_dense()

        with tf.GradientTape() as tape:
            x_tf = tf.convert_to_tensor(x_input)
            y_tf = tf.convert_to_tensor(y_true)

            tape.watch(x_tf)
            y_pred = model(x_tf)
            loss = tf.keras.losses.categorical_crossentropy(y_tf, y_pred)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables + [x_tf])
        keras_dkernel = grads[0].numpy()  # Wax.T
        keras_drecurrent = grads[1].numpy()  # Waa.T
        keras_dbias = grads[2].numpy()

        keras_dWya = grads[3].numpy()
        keras_dby = grads[4].numpy()
        keras_dx = grads[5].numpy().transpose(2, 0, 1)  # (n_x, m, T_x)

        # Your gradients
        my_grads, _ = rnn_backward(self.X, self.Y, self.parameters, self.cache)
        # Scale by number of timesteps to match Keras' default reduction
        T = self.T_x
        for key in ["dWax", "dWaa", "dba", "dWya", "dby"]:
            my_grads[key] /= T

        print_matrix_diff(my_grads["dWax"], keras_dkernel.T, name="dWax")
        # Compare
        np.testing.assert_allclose(
            my_grads["dWax"], keras_dkernel.T, atol=1e-3, err_msg="Mismatch in dWax"
        )

        print_matrix_diff(my_grads["dWaa"], keras_drecurrent.T, name="dWaa")

        np.testing.assert_allclose(
            my_grads["dWaa"],
            keras_drecurrent.T,
            atol=3.2e-3,
            err_msg="Mismatch in dWaa",
        )

        print_matrix_diff(my_grads["dba"], keras_dbias.T, name="dba")
        np.testing.assert_allclose(
            my_grads["dba"].flatten(), keras_dbias, atol=1e-3, err_msg="Mismatch in dba"
        )

        diff = np.abs(my_grads["dWya"] - keras_dWya.T)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print("Largest diff at:", max_diff_idx)
        print("From-scratch:", my_grads["dWya"][max_diff_idx])
        print("Keras       :", keras_dWya.T[max_diff_idx])
        print("Abs diff    :", diff[max_diff_idx])

        print_matrix_diff(my_grads["dWya"], keras_dWya.T, name="dWya")
        np.testing.assert_allclose(
            my_grads["dWya"], keras_dWya.T, atol=4.0e-3, err_msg="Mismatch in dWya"
        )
        np.testing.assert_allclose(
            my_grads["dby"].flatten(), keras_dby, atol=1e-3, err_msg="Mismatch in dby"
        )

        print(
            "\n✅ Passed: rnn_backward matches Keras gradients (softmax + categorical crossentropy)"
        )

    def test_per_timestep_loss_debug(self):
        """
        🧪 Debug-only test to inspect per-timestep loss values from Keras.

        This test does NOT assert any correctness. Instead, it helps you:
        - Understand how each timestep contributes to the total loss
        - Spot potential irregularities across time
        - Verify alignment with your manual loss computations

        Why it's helpful:
        - Keras default reduction (mean) can hide per-timestep mismatches
        - Use this as a sanity-checking tool when tuning the RNN

        Note:
        - `reduction='none'` returns shape (batch_size, T_x)
        - Use tf.reduce_mean() or tf.reduce_sum() manually if needed
        """
        model = self.keras_rnn_with_dense()
        x_input = np.stack([np.eye(self.vocab_size)[self.X]] * self.m).astype(
            np.float32
        )
        y_true = np.stack([np.eye(self.vocab_size)[self.Y]] * self.m).astype(np.float32)

        x_tf = tf.convert_to_tensor(x_input)
        y_tf = tf.convert_to_tensor(y_true)

        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            y_pred = model(x_tf)

            # No reduction – keep per-timestep losses for inspection
            loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction="none")
            per_timestep_loss = loss_fn(y_tf, y_pred)

        print("\n🧩 Per-timestep loss values:")
        print(per_timestep_loss.numpy())  # Shape: (batch_size, T_x)

        # Optional: manually reduce if you want a scalar loss value
        total_loss = tf.reduce_mean(per_timestep_loss)
        print("🧮 Mean loss:", total_loss.numpy())

        # You can also inspect gradients if needed
        # grads = tape.gradient(total_loss, model.trainable_variables)


if __name__ == "__main__":
    unittest.main()
