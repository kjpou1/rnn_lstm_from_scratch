# test_numerical_grad_check.py
# ----------------------------------------------------
# This unit test verifies the correctness of your backpropagation implementation
# for key LSTM gradients using numerical gradient checking.
#
# Each test compares analytical gradients from `lstm_backwards()` with finite-difference
# approximations to ensure consistency. This helps catch subtle bugs in the backward
# pass related to broadcasting, slicing, or gate interactions.
#
# Tests:
#   ✅ dWf – weight matrix of forget gate
#   ✅ dbf – bias vector of forget gate
#   ✅ dx  – input sequence gradient
# ----------------------------------------------------

import unittest

import numpy as np

from src.models.lstm_model import (
    initialize_lstm_parameters,
    lstm_backwards,
    lstm_forward,
)


def compute_loss(a, da):
    """
    Simulate a scalar loss by dotting the output activations
    with upstream gradient (e.g., ∑ a * da).
    """
    return np.sum(a * da)


def numerical_gradient(param, param_name, parameters, x, a0, da, epsilon=1e-5):
    """
    Numerically approximate the gradient of a parameter via centered difference.

    Args:
        param: Parameter matrix (e.g. Wf, bf)
        param_name: Key in the parameter dictionary (e.g. 'Wf')
        parameters: Dictionary of all model parameters
        x: Input data (n_x, m, T_x)
        a0: Initial hidden state (n_a, m)
        da: Upstream gradient (n_a, m, T_x)
        epsilon: Small value to perturb for finite differences

    Returns:
        grad_approx: Array of same shape as param with numerical gradients
    """
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

        # Restore original value
        param[idx] = original_value
        it.iternext()

    return grad_approx


class TestNumericalGradientCheck(unittest.TestCase):
    def test_numerical_vs_backprop_dWf(self):
        """
        🔍 Checks dWf (forget gate weights).
        This is the most sensitive and representative gate gradient.
        """
        np.random.seed(5)
        n_x, n_a, n_y, m, T_x = 3, 5, 2, 4, 5

        x = np.random.randn(n_x, m, T_x)
        a0 = np.random.randn(n_a, m)
        da = np.random.randn(n_a, m, T_x)

        parameters = initialize_lstm_parameters(n_a, n_x, n_y)
        _, _, caches = lstm_forward(x, a0, parameters)
        analytical_grads = lstm_backwards(da, (caches, x))
        dWf_analytical = analytical_grads["dWf"]

        dWf_numeric = numerical_gradient(
            parameters["Wf"].copy(), "Wf", parameters.copy(), x, a0, da
        )

        np.testing.assert_allclose(
            dWf_analytical, dWf_numeric, atol=1e-5, err_msg="Mismatch in dWf gradient"
        )
        print("✅ Numerical gradient check passed for dWf")

    def test_numerical_vs_backprop_dbf(self):
        """
        🔍 Checks dbf (bias for forget gate).
        Ensures bias gradients are correctly accumulated over the batch.
        """
        np.random.seed(6)
        n_x, n_a, n_y, m, T_x = 3, 5, 2, 4, 5

        x = np.random.randn(n_x, m, T_x)
        a0 = np.random.randn(n_a, m)
        da = np.random.randn(n_a, m, T_x)

        parameters = initialize_lstm_parameters(n_a, n_x, n_y)
        _, _, caches = lstm_forward(x, a0, parameters)
        analytical_grads = lstm_backwards(da, (caches, x))
        dbf_analytical = analytical_grads["dbf"]

        dbf_numeric = numerical_gradient(
            parameters["bf"].copy(), "bf", parameters.copy(), x, a0, da
        )

        np.testing.assert_allclose(
            dbf_analytical, dbf_numeric, atol=1e-5, err_msg="Mismatch in dbf gradient"
        )
        print("✅ Numerical gradient check passed for dbf")

    def test_numerical_vs_backprop_dx(self):
        """
        🔍 Checks dx (gradient w.r.t input sequence).
        This verifies whether the gradients correctly flow into the input.
        """
        np.random.seed(7)
        n_x, n_a, n_y, m, T_x = 3, 5, 2, 2, 3  # smaller test for speed

        x = np.random.randn(n_x, m, T_x)
        a0 = np.random.randn(n_a, m)
        da = np.random.randn(n_a, m, T_x)

        parameters = initialize_lstm_parameters(n_a, n_x, n_y)
        _, _, caches = lstm_forward(x, a0, parameters)
        analytical_grads = lstm_backwards(da, (caches, x))
        dx_analytical = analytical_grads["dx"]

        dx_numeric = np.zeros_like(x)
        epsilon = 1e-5

        for i in range(n_x):
            for j in range(m):
                for t in range(T_x):
                    original = x[i, j, t]

                    x[i, j, t] = original + epsilon
                    a_plus, _, _ = lstm_forward(x, a0, parameters)
                    loss_plus = compute_loss(a_plus, da)

                    x[i, j, t] = original - epsilon
                    a_minus, _, _ = lstm_forward(x, a0, parameters)
                    loss_minus = compute_loss(a_minus, da)

                    dx_numeric[i, j, t] = (loss_plus - loss_minus) / (2 * epsilon)
                    x[i, j, t] = original  # restore

        np.testing.assert_allclose(
            dx_analytical, dx_numeric, atol=1e-5, err_msg="Mismatch in dx gradient"
        )
        print("✅ Numerical gradient check passed for dx")


if __name__ == "__main__":
    unittest.main()
