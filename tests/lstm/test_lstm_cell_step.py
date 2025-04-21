import unittest

import numpy as np

from src.models.lstm_model import lstm_cell_step
from src.utils.utils import softmax  # adjust path if needed


class TestLSTMCellStep(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        # Dimensions
        self.n_x = 3
        self.n_a = 5
        self.n_y = 2
        self.m = 10

        # Inputs
        self.xt = np.random.randn(self.n_x, self.m)
        self.a_prev = np.random.randn(self.n_a, self.m)
        self.c_prev = np.random.randn(self.n_a, self.m)

        # Parameters
        self.parameters = {
            "Wf": np.random.randn(self.n_a, self.n_a + self.n_x),
            "bf": np.random.randn(self.n_a, 1),
            "Wi": np.random.randn(self.n_a, self.n_a + self.n_x),
            "bi": np.random.randn(self.n_a, 1),
            "Wo": np.random.randn(self.n_a, self.n_a + self.n_x),
            "bo": np.random.randn(self.n_a, 1),
            "Wc": np.random.randn(self.n_a, self.n_a + self.n_x),
            "bc": np.random.randn(self.n_a, 1),
            "Wy": np.random.randn(self.n_y, self.n_a),
            "by": np.random.randn(self.n_y, 1),
        }

    def test_lstm_cell_forward_shapes(self):
        a_next, c_next, logits, cache = lstm_cell_step(
            self.xt, self.a_prev, self.c_prev, self.parameters
        )

        self.assertEqual(a_next.shape, (self.n_a, self.m), "Incorrect shape for a_next")
        self.assertEqual(c_next.shape, (self.n_a, self.m), "Incorrect shape for c_next")
        self.assertEqual(logits.shape, (self.n_y, self.m), "Incorrect shape for logits")
        self.assertIsInstance(cache, tuple, "Cache must be a tuple")
        self.assertEqual(len(cache), 10, "Cache tuple must contain 10 elements")

    def test_lstm_cell_step_values(self):
        a_next, c_next, logits, _ = lstm_cell_step(
            self.xt, self.a_prev, self.c_prev, self.parameters
        )

        y_hat = softmax(logits)

        # ---- Logging ----
        print("\n[DEBUG] LSTM Cell Output")
        print("-" * 40)
        print("Logits ([:, 0]):")
        print(logits[:, 0])
        print("\nSoftmax Probabilities ([:, 0]):")
        print(y_hat[:, 0])
        print("\na_next[4, 0]:", a_next[4, 0])
        print("c_next[2, 0]:", c_next[2, 0])
        print("-" * 40)

        # ---- Assertions ----
        np.testing.assert_almost_equal(
            a_next[4, 0],
            -0.664085,
            decimal=5,
            err_msg="Unexpected value in a_next[4,0]",
        )
        np.testing.assert_almost_equal(
            c_next[2, 0],
            0.632678,
            decimal=5,
            err_msg="Unexpected value in c_next[2,0]",
        )
        np.testing.assert_almost_equal(
            logits[1, 0],
            0.290429,
            decimal=5,
            err_msg="Unexpected value in logits[1,0]",
        )
        np.testing.assert_almost_equal(
            y_hat[1, 0],
            0.799139,
            decimal=5,
            err_msg="Unexpected value in y_hat[1,0]",
        )


if __name__ == "__main__":
    unittest.main()
