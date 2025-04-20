import unittest

import numpy as np

from src.models.lstm_model import (
    initialize_lstm_parameters,
    lstm_cell_step,
    lstm_step_backward,
)


class TestLSTMStepBackward(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        self.n_x = 3
        self.n_a = 5
        self.n_y = 2
        self.m = 10

        # Inputs
        self.xt = np.random.randn(self.n_x, self.m)
        self.a_prev = np.random.randn(self.n_a, self.m)
        self.c_prev = np.random.randn(self.n_a, self.m)

        # Parameters
        self.parameters = initialize_lstm_parameters(
            self.n_a, self.n_x, self.n_y, seed=1
        )

        # Forward pass to get cache
        self.a_next, self.c_next, self.yt_pred, self.cache = lstm_cell_step(
            self.xt, self.a_prev, self.c_prev, self.parameters
        )

    def test_shapes_of_gradients(self):
        # Fake upstream gradients from next time step
        da_next = np.random.randn(self.n_a, self.m)
        dc_next = np.random.randn(self.n_a, self.m)

        grads = lstm_step_backward(da_next, dc_next, self.cache)

        self.assertEqual(grads["dxt"].shape, (self.n_x, self.m))
        self.assertEqual(grads["da_prev"].shape, (self.n_a, self.m))
        self.assertEqual(grads["dc_prev"].shape, (self.n_a, self.m))

        for name in ["dWf", "dWi", "dWc", "dWo"]:
            self.assertEqual(
                grads[name].shape,
                (self.n_a, self.n_a + self.n_x),
                f"{name} shape mismatch",
            )

        for name in ["dbf", "dbi", "dbc", "dbo"]:
            self.assertEqual(grads[name].shape, (self.n_a, 1), f"{name} shape mismatch")

    def test_print_sample_values(self):
        # Optional: just for inspection
        da_next = np.random.randn(self.n_a, self.m)
        dc_next = np.random.randn(self.n_a, self.m)

        grads = lstm_step_backward(da_next, dc_next, self.cache)

        print("Sample gradients:")
        print("dxt[0,0] =", grads["dxt"][0, 0])
        print("da_prev[0,0] =", grads["da_prev"][0, 0])
        print("dc_prev[0,0] =", grads["dc_prev"][0, 0])
        print("dWf[0,0] =", grads["dWf"][0, 0])
        print("dbf[0,0] =", grads["dbf"][0, 0])


if __name__ == "__main__":
    unittest.main()
