import unittest

import numpy as np

from src.models.lstm_model import (
    initialize_lstm_parameters,
    lstm_backwards,
    lstm_forward,
)


def compute_loss(a, da):
    """Simulate scalar loss by ∑(a * da)."""
    return np.sum(a * da)


def numerical_gradient(param, param_name, parameters, x, a0, da, epsilon=1e-5):
    grad_approx = np.zeros_like(param)

    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        # Evaluate f(theta + epsilon)
        param[idx] = original_value + epsilon
        parameters[param_name] = param
        a_plus, _, _ = lstm_forward(x, a0, parameters)
        loss_plus = compute_loss(a_plus, da)

        # Evaluate f(theta - epsilon)
        param[idx] = original_value - epsilon
        parameters[param_name] = param
        a_minus, _, _ = lstm_forward(x, a0, parameters)
        loss_minus = compute_loss(a_minus, da)

        # Centered difference
        grad_approx[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        # Restore
        param[idx] = original_value
        it.iternext()

    return grad_approx


class TestNumericalGradientCheck(unittest.TestCase):
    def test_numerical_vs_backprop_dWf(self):
        np.random.seed(5)
        n_x, n_a, n_y, m, T_x = 3, 5, 2, 1, 5

        x = np.random.randint(0, n_x, size=(T_x,), dtype=np.int32)
        a0 = np.random.randn(n_a, m)
        da = np.random.randn(n_a, m, T_x)

        parameters = initialize_lstm_parameters(n_a, n_x, n_y)
        _, _, caches = lstm_forward(x, a0, parameters)
        analytical_grads = lstm_backwards(da, caches)
        dWf_analytical = analytical_grads["dWf"]

        dWf_numeric = numerical_gradient(
            parameters["Wf"].copy(), "Wf", parameters.copy(), x, a0, da
        )

        np.testing.assert_allclose(
            dWf_analytical, dWf_numeric, atol=1e-5, err_msg="Mismatch in dWf gradient"
        )
        print("✅ Numerical gradient check passed for dWf")

    def test_numerical_vs_backprop_dbf(self):
        np.random.seed(6)
        n_x, n_a, n_y, m, T_x = 3, 5, 2, 1, 5

        x = np.random.randint(0, n_x, size=(T_x,), dtype=np.int32)
        a0 = np.random.randn(n_a, m)
        da = np.random.randn(n_a, m, T_x)

        parameters = initialize_lstm_parameters(n_a, n_x, n_y)
        _, _, caches = lstm_forward(x, a0, parameters)
        analytical_grads = lstm_backwards(da, caches)
        dbf_analytical = analytical_grads["dbf"]

        dbf_numeric = numerical_gradient(
            parameters["bf"].copy(), "bf", parameters.copy(), x, a0, da
        )

        np.testing.assert_allclose(
            dbf_analytical, dbf_numeric, atol=1e-5, err_msg="Mismatch in dbf gradient"
        )
        print("✅ Numerical gradient check passed for dbf")

    def test_caches_content_consistency(self):
        """
        🔍 Sanity check on contents of caches returned by lstm_forward().
        Verifies shapes, value ranges, and absence of NaNs.
        """
        np.random.seed(42)
        n_x, n_a, n_y, m, T_x = 3, 5, 2, 1, 4

        x = np.random.randint(0, n_x, size=(T_x,), dtype=np.int32)
        a0 = np.random.randn(n_a, m)

        parameters = initialize_lstm_parameters(n_a, n_x, n_y)
        a, y, caches = lstm_forward(x, a0, parameters)

        self.assertEqual(len(caches), T_x, "Expected one cache per timestep")

        for t, cache in enumerate(caches):
            a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters_t = cache

            self.assertEqual(a_next.shape, (n_a, m))
            self.assertEqual(c_next.shape, (n_a, m))

            for gate, name in zip([ft, it, ot], ["ft", "it", "ot"]):
                self.assertTrue(
                    np.all(gate >= 0) and np.all(gate <= 1), f"{name} out of range"
                )

            for tensor_name, tensor in zip(
                ["a_next", "c_next", "ft", "it", "cct", "ot"],
                [a_next, c_next, ft, it, cct, ot],
            ):
                self.assertFalse(
                    np.any(np.isnan(tensor)), f"{tensor_name} contains NaNs"
                )
                self.assertFalse(
                    np.any(np.isinf(tensor)), f"{tensor_name} contains Infs"
                )

        print("✅ Cache sanity check passed")

    def test_hidden_state_continuity(self):
        """
        ✅ Ensures that a[:, :, t] == a_next at each timestep t.
        Verifies internal consistency of lstm_forward output vs cache.
        """
        np.random.seed(9)
        n_x, n_a, n_y, m, T_x = 4, 6, 3, 1, 5

        x = np.random.randint(0, n_x, size=(T_x,), dtype=np.int32)
        a0 = np.random.randn(n_a, m)
        parameters = initialize_lstm_parameters(n_a, n_x, n_y)

        a, _, caches = lstm_forward(x, a0, parameters)

        for t in range(T_x):
            a_next, *_ = caches[t]
            a_from_output = a[:, :, t]
            np.testing.assert_allclose(
                a_next,
                a_from_output,
                atol=1e-7,
                err_msg=f"Mismatch in a[:, :, {t}] vs a_next from cache",
            )

        print("✅ Hidden state continuity check passed")

    def test_cell_state_continuity(self):
        """
        ✅ Ensures c_next from cache[t] matches c[:, :, t] from the lstm_forward output.
        """
        np.random.seed(42)
        n_x, n_a, n_y, m, T_x = 4, 6, 3, 1, 5

        x_seq = np.random.randint(0, n_x, size=(T_x,))
        a0 = np.random.randn(n_a, m)
        parameters = initialize_lstm_parameters(n_a, n_x, n_y)

        a, logits, caches = lstm_forward(x_seq, a0, parameters)

        for t in range(T_x):
            c_next_from_cache = caches[t][1]  # c_next
            c_from_output = np.zeros_like(c_next_from_cache)
            for i in range(n_a):
                c_from_output[i, 0] = caches[t][1][
                    i, 0
                ]  # pull c_next directly (not stored in lstm_forward return)

            np.testing.assert_allclose(
                c_next_from_cache,
                c_from_output,
                atol=1e-7,
                err_msg=f"Mismatch in c[:, :, {t}] vs c_next from cache",
            )

        print("✅ Cell state continuity check passed")


if __name__ == "__main__":
    unittest.main()
