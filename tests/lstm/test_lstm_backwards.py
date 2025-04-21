import unittest

import numpy as np

from src.models.lstm_model import (
    initialize_lstm_parameters,
    lstm_backwards,
    lstm_forward,
)


class TestLSTMBackwards(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

        self.vocab_size = 6
        self.n_a = 5
        self.n_y = 2
        self.T_x = 7

        # Token indices (batch size 1)
        self.x_seq = np.random.randint(
            0, self.vocab_size, size=(self.T_x,), dtype=np.int32
        )
        self.a0 = np.random.randn(self.n_a, 1)

        self.parameters = initialize_lstm_parameters(
            self.n_a, self.vocab_size, self.n_y, seed=2
        )

        self.a, self.y, self.caches = lstm_forward(self.x_seq, self.a0, self.parameters)

        self.da = np.random.randn(self.n_a, 1, self.T_x)  # shape (n_a, 1, T_x)

    def test_shapes_of_gradients(self):
        grads = lstm_backwards(self.da, self.caches)

        # self.assertEqual(grads["dx"].shape, (self.T_x,))
        self.assertEqual(grads["da0"].shape, (self.n_a, 1))

        for name in ["dWf", "dWi", "dWc", "dWo"]:
            self.assertEqual(
                grads[name].shape,
                (self.n_a, self.n_a + self.vocab_size),
                f"{name} shape mismatch",
            )

        for name in ["dbf", "dbi", "dbc", "dbo"]:
            self.assertEqual(grads[name].shape, (self.n_a, 1), f"{name} shape mismatch")

    def test_print_sample_values(self):
        grads = lstm_backwards(self.da, self.caches)

        print("\n🔍 Sample values from lstm_backwards:")
        # print("dx[0] =", grads["dx"][0])
        print("da0[0,0] =", grads["da0"][0, 0])
        print("dWf[0,0] =", grads["dWf"][0, 0])
        print("dbf[0,0] =", grads["dbf"][0, 0])

    def test_gradient_accumulation_consistency(self):
        from src.models.lstm_model import lstm_step_backward

        grads_seq = lstm_backwards(self.da, self.caches)

        # Init accumulators
        dWf_acc = np.zeros_like(self.parameters["Wf"])
        dWi_acc = np.zeros_like(self.parameters["Wi"])
        dWc_acc = np.zeros_like(self.parameters["Wc"])
        dWo_acc = np.zeros_like(self.parameters["Wo"])
        dbf_acc = np.zeros_like(self.parameters["bf"])
        dbi_acc = np.zeros_like(self.parameters["bi"])
        dbc_acc = np.zeros_like(self.parameters["bc"])
        dbo_acc = np.zeros_like(self.parameters["bo"])

        da_prevt = np.zeros((self.n_a, 1))
        dc_prevt = np.zeros((self.n_a, 1))

        for t in reversed(range(self.T_x)):
            da_curr = self.da[:, :, t] + da_prevt
            dc_curr = dc_prevt

            step_grads = lstm_step_backward(da_curr, dc_curr, self.caches[t])

            dWf_acc += step_grads["dWf"]
            dWi_acc += step_grads["dWi"]
            dWc_acc += step_grads["dWc"]
            dWo_acc += step_grads["dWo"]
            dbf_acc += step_grads["dbf"]
            dbi_acc += step_grads["dbi"]
            dbc_acc += step_grads["dbc"]
            dbo_acc += step_grads["dbo"]

            da_prevt = step_grads["da_prev"]
            dc_prevt = step_grads["dc_prev"]

        # Assert match with full-sequence backprop
        np.testing.assert_allclose(dWf_acc, grads_seq["dWf"], atol=1e-6)
        np.testing.assert_allclose(dWi_acc, grads_seq["dWi"], atol=1e-6)
        np.testing.assert_allclose(dWc_acc, grads_seq["dWc"], atol=1e-6)
        np.testing.assert_allclose(dWo_acc, grads_seq["dWo"], atol=1e-6)
        np.testing.assert_allclose(dbf_acc, grads_seq["dbf"], atol=1e-6)
        np.testing.assert_allclose(dbi_acc, grads_seq["dbi"], atol=1e-6)
        np.testing.assert_allclose(dbc_acc, grads_seq["dbc"], atol=1e-6)
        np.testing.assert_allclose(dbo_acc, grads_seq["dbo"], atol=1e-6)

        print("\n✅ Per-step accumulation matches full lstm_backwards")


if __name__ == "__main__":
    unittest.main()
