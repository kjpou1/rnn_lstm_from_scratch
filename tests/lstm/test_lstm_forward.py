# tests/lstm/test_lstm_forward.py

import unittest

import numpy as np

from src.models.lstm_model import initialize_lstm_parameters, lstm_forward
from src.utils.utils import softmax


class TestLSTMForward(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        # Sequence & model config
        self.T_x = 4
        self.vocab_size = 6
        self.n_a = 5
        self.n_y = 2

        # Simulated token indices (1D array)
        self.x_seq = np.array([1, 3, 2, 5], dtype=np.int32)  # shape (T_x,)
        self.a0 = np.random.randn(self.n_a, 1)

        self.parameters = initialize_lstm_parameters(
            self.n_a, self.vocab_size, self.n_y
        )

        # Run forward
        self.a, self.y_logits, _ = lstm_forward(self.x_seq, self.a0, self.parameters)

        # print("new expected_a_final =", self.a[:, 0, -1])
        # print("new expected_logits_final =", self.y_logits[:, 0, -1])
        # print("new expected_softmax_final =", softmax(self.y_logits)[:, 0, -1])

        # Reference outputs (final timestep)
        self.expected_a_final = np.array(
            [
                [-1.89051411e-03],
                [-3.13858351e-03],
                [1.99142448e-03],
                [3.81602733e-03],
                [-3.33245921e-05],
            ]
        )
        self.expected_logits_final = np.array(
            [
                [-1.07238560e-05],
                [-5.97954059e-05],
            ]
        )
        self.expected_softmax_final = np.array(
            [
                [0.50001227],
                [0.49998773],
            ]
        )

        print("\n🔧 LSTM Forward (Index-Based) Setup Complete")
        print(f"  - x_seq: {self.x_seq}")
        print(f"  - Output shape a: {self.a.shape}")
        print(f"  - Output shape y: {self.y_logits.shape}")

    def test_lstm_forward_shapes(self):
        print("\n✅ Checking output shapes...")
        self.assertEqual(self.a.shape, (self.n_a, 1, self.T_x))
        self.assertEqual(self.y_logits.shape, (self.n_y, 1, self.T_x))
        print("✔️ Passed shape check")

    def test_lstm_forward_softmax_values(self):
        print("\n🔍 Validating softmax output at final time step...")
        y_softmax = softmax(self.y_logits)
        actual = y_softmax[:, :, -1]
        expected = self.expected_softmax_final
        print("🔹 Final softmax y[:, :, -1]:")
        print(actual)

        np.testing.assert_almost_equal(
            actual,
            expected,
            decimal=6,
            err_msg="Mismatch in final softmax prediction y[:, :, -1]",
        )

    def test_lstm_forward_hidden_state_elementwise(self):
        print("\n🔎 Detailed comparison for a[:, 0, -1]...")
        a_final = self.a[:, 0, -1].reshape(-1, 1)
        mismatches = np.sum(
            ~np.isclose(a_final, self.expected_a_final, rtol=1e-5, atol=1e-7)
        )

        if mismatches == 0:
            print("✔️ All values match in final hidden state")
        else:
            print(f"❌ Mismatches found in final hidden state: {mismatches}")

        np.testing.assert_almost_equal(
            a_final,
            self.expected_a_final,
            decimal=6,
            err_msg="Mismatch in final hidden state a[:, :, -1]",
        )

    def test_lstm_forward_logits_values(self):
        print("\n🧪 Validating raw logits at final time step...")
        actual = self.y_logits[:, :, -1]
        expected = self.expected_logits_final
        print("🔹 Final logits y[:, :, -1]:")
        print(actual)

        np.testing.assert_almost_equal(
            actual,
            expected,
            decimal=6,
            err_msg="Mismatch in final logits y[:, :, -1]",
        )


if __name__ == "__main__":
    unittest.main()
