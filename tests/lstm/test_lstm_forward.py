# tests/lstm/test_lstm_forward.py
import unittest

import numpy as np

from src.models.lstm_model import initialize_lstm_parameters, lstm_forward


class TestLSTMForward(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

        # Dimensions
        self.n_x, self.m, self.T_x = 3, 4, 3
        self.n_a, self.n_y = 5, 2

        # Test data
        self.x = np.random.randn(self.n_x, self.m, self.T_x)
        self.a0 = np.random.randn(self.n_a, self.m)
        self.parameters = initialize_lstm_parameters(self.n_a, self.n_x, self.n_y)
        self.a, self.y, _ = lstm_forward(self.x, self.a0, self.parameters)

        # Expected values
        self.expected_a_final = np.array(
            [
                [0.0030695, -0.0001014, 0.0003386, 0.0013613],
                [-0.0044756, -0.0014147, -0.0004756, -0.0005169],
                [-0.0014000, -0.0012035, 0.0029781, -0.0055609],
                [0.0075187, 0.0026621, -0.0008817, 0.0046018],
                [-0.0122554, -0.0014587, -0.0000641, -0.0021135],
            ]
        )
        self.expected_y_final = np.array(
            [
                [0.4998994, 0.4999817, 0.5000074, 0.4999597],
                [0.5001006, 0.5000183, 0.4999926, 0.5000403],
            ]
        )

        print("\n🔧 LSTM Forward Setup Complete")
        print(f"  - Input shape x: {self.x.shape}")
        print(f"  - Output shape a: {self.a.shape}, dtype: {self.a.dtype}")
        print(f"  - Output shape y: {self.y.shape}, dtype: {self.y.dtype}")

    def test_lstm_forward_shapes(self):
        print("\n✅ Checking output shapes...")
        self.assertEqual(self.a.shape, (self.n_a, self.m, self.T_x))
        self.assertEqual(self.y.shape, (self.n_y, self.m, self.T_x))
        print("✔️ Passed shape check")

    def test_lstm_forward_values(self):
        print("\n🔍 Validating LSTM final outputs...")
        print("🔹 Final a[:, :, -1]:")
        print(np.round(self.a[:, :, -1], 7))
        print("🔹 Final y[:, :, -1]:")
        print(np.round(self.y[:, :, -1], 7))

        np.testing.assert_almost_equal(
            self.a[:, :, -1],
            self.expected_a_final,
            decimal=6,
            err_msg="Mismatch in final hidden state a[:, :, -1]",
        )
        np.testing.assert_almost_equal(
            self.y[:, :, -1],
            self.expected_y_final,
            decimal=6,
            err_msg="Mismatch in final prediction y[:, :, -1]",
        )
        print("✔️ Final values match expected")

    def test_lstm_forward_hidden_state_elementwise(self):
        print("\n🔎 Detailed elementwise comparison for a[:, :, -1]")
        a_final = self.a[:, :, -1]
        total_mismatches = 0

        for i in range(self.expected_a_final.shape[0]):
            for j in range(self.expected_a_final.shape[1]):
                actual = a_final[i, j]
                expected = self.expected_a_final[i, j]
                if not np.isclose(actual, expected, rtol=1e-5, atol=1e-7):
                    print(
                        f"❌ Mismatch at a[{i},{j}]: expected {expected}, got {actual}"
                    )
                    total_mismatches += 1

        if total_mismatches == 0:
            print("✔️ All values match in final hidden state")
        else:
            print(f"❗ Total mismatches: {total_mismatches}")

        self.assertEqual(
            total_mismatches,
            0,
            f"Total mismatches in final hidden state: {total_mismatches}",
        )


if __name__ == "__main__":
    unittest.main()
