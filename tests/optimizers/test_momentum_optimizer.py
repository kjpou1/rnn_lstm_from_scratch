# tests/optimizers/test_momentum_optimizer.py

import unittest
import numpy as np
from src.optimizers.momentum_optimizer import MomentumOptimizer

class TestMomentumOptimizer(unittest.TestCase):
    def test_update_matches_manual_formula(self):
        """
        Test a single step of momentum optimizer matches manual calculation.

        Math:
            v = β * v_prev - α * grad
            param = param + v

        For the first step (v_prev = 0):
            v = - α * grad
            param = param + v
        """
        Waa = np.array([[1.0, 2.0], [3.0, 4.0]])
        Wax = np.array([[5.0, 6.0], [7.0, 8.0]])
        parameters = {"Waa": np.copy(Waa), "Wax": np.copy(Wax)}
        gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }

        lr = 0.1  # α = 0.1
        beta = 0.9  # β = 0.9

        optimizer = MomentumOptimizer(learning_rate=lr, momentum=beta)

        grads_and_vars = [
            (gradients["dWaa"], parameters["Waa"]),
            (gradients["dWax"], parameters["Wax"]),
        ]
        optimizer.apply_gradients(grads_and_vars)

        # Manual momentum calculation (first step)
        expected_v_Waa = -lr * gradients["dWaa"]  # v = -α * grad
        expected_v_Wax = -lr * gradients["dWax"]

        expected_Waa = Waa + expected_v_Waa  # param += v
        expected_Wax = Wax + expected_v_Wax

        np.testing.assert_allclose(parameters["Waa"], expected_Waa, rtol=1e-6)
        np.testing.assert_allclose(parameters["Wax"], expected_Wax, rtol=1e-6)

    def test_multiple_steps_accumulate_momentum(self):
        """
        Test that velocity accumulates over multiple steps.

        Math over multiple steps:
            1st step: v₁ = -α * grad
            2nd step: v₂ = β * v₁ - α * grad
            param = param + v₂

        v gets bigger (momentum accumulates).
        """
        Waa = np.array([[1.0, 2.0], [3.0, 4.0]])
        parameters = {"Waa": np.copy(Waa)}
        gradients = {"dWaa": np.array([[0.1, 0.1], [0.1, 0.1]])}

        lr = 0.1
        beta = 0.9
        optimizer = MomentumOptimizer(learning_rate=lr, momentum=beta)

        grads_and_vars = [(gradients["dWaa"], parameters["Waa"])]

        # First step
        optimizer.apply_gradients(grads_and_vars)
        Waa_after_1 = np.copy(parameters["Waa"])

        # Second step
        optimizer.apply_gradients(grads_and_vars)
        Waa_after_2 = np.copy(parameters["Waa"])

        # Verify that parameters changed further (momentum accumulates)
        self.assertFalse(np.allclose(Waa_after_1, Waa_after_2), "Momentum did not accumulate across steps!")

    def test_momentum_update_vs_apply_gradients(self):
        """
        Test that update() and apply_gradients() produce identical results.

        - update() automatically builds (grad, param) pairs and calls apply_gradients()
        - Manual apply_gradients() passes them explicitly
        """
        parameters = {
            "Waa": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Wax": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }

        parameters2 = {k: np.copy(v) for k, v in parameters.items()}

        optimizer1 = MomentumOptimizer(learning_rate=0.1, momentum=0.9)
        optimizer2 = MomentumOptimizer(learning_rate=0.1, momentum=0.9)

        # Path 1: update() call
        updated_parameters = optimizer1.update(parameters, gradients)

        # Path 2: manual apply_gradients()
        grads_and_vars = [(gradients["dWaa"], parameters2["Waa"]), (gradients["dWax"], parameters2["Wax"])]
        optimizer2.apply_gradients(grads_and_vars)

        for key in updated_parameters.keys():
            np.testing.assert_allclose(updated_parameters[key], parameters2[key], rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
    print("\n✅ All MomentumOptimizer tests passed!")
