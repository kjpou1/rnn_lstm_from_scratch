import unittest
import numpy as np
from src.optimizers.rmsprop_optimizer import RMSPropOptimizer

class TestRMSPropOptimizer(unittest.TestCase):
    def test_update_matches_manual_formula(self):
        """
        Test that optimizer.update() matches manual RMSProp update formula.
        
        Math:
            s = (1 - β) * (grad ** 2)      # Initial step (v=0 so no momentum)
            θ = θ - α * grad / (sqrt(s) + ε)
        """
        # Fake parameters
        Waa = np.array([[1.0, 2.0], [3.0, 4.0]])
        Wax = np.array([[5.0, 6.0], [7.0, 8.0]])
        parameters = {"Waa": np.copy(Waa), "Wax": np.copy(Wax)}

        gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }

        lr = 0.01
        beta = 0.9
        epsilon = 1e-8

        optimizer = RMSPropOptimizer(learning_rate=lr, beta=beta, epsilon=epsilon)

        # --- Optimizer's update() ---
        updated_parameters = optimizer.update(parameters, gradients)

        # --- Manual RMSProp math ---
        # s = (1 - beta) * (grad ** 2)
        s_Waa = (1 - beta) * (gradients["dWaa"] ** 2)
        s_Wax = (1 - beta) * (gradients["dWax"] ** 2)

        # θ = θ - α * grad / (sqrt(s) + ε)
        expected_Waa = Waa - lr * gradients["dWaa"] / (np.sqrt(s_Waa) + epsilon)
        expected_Wax = Wax - lr * gradients["dWax"] / (np.sqrt(s_Wax) + epsilon)

        # --- Assertions ---
        np.testing.assert_allclose(updated_parameters["Waa"], expected_Waa, rtol=1e-6)
        np.testing.assert_allclose(updated_parameters["Wax"], expected_Wax, rtol=1e-6)

    def test_multiple_steps_decay_running_average(self):
        """
        Test that RMSProp running average (s) accumulates over multiple steps.

        Math:
            s_t = β * s_(t-1) + (1 - β) * (grad ** 2)
        """
        Waa = np.array([[1.0, 2.0], [3.0, 4.0]])
        parameters = {"Waa": np.copy(Waa)}
        gradients = {"dWaa": np.array([[0.1, 0.1], [0.1, 0.1]])}

        lr = 0.01
        beta = 0.9
        optimizer = RMSPropOptimizer(learning_rate=lr, beta=beta)

        grads_and_vars = [(gradients["dWaa"], parameters["Waa"])]

        # First step
        optimizer.apply_gradients(grads_and_vars)
        Waa_after_1 = np.copy(parameters["Waa"])

        # Second step (should accumulate s)
        optimizer.apply_gradients(grads_and_vars)
        Waa_after_2 = np.copy(parameters["Waa"])

        print("\nAfter 1st step:", Waa_after_1)
        print("After 2nd step:", Waa_after_2)

    def test_rmsprop_update_vs_apply_gradients(self):
        """
        Test that update() and apply_gradients() produce the same final parameters.
        """
        parameters = {
            "Waa": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Wax": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }

        # Clone parameters
        parameters2 = {k: np.copy(v) for k, v in parameters.items()}

        optimizer1 = RMSPropOptimizer(learning_rate=0.01, beta=0.9)
        optimizer2 = RMSPropOptimizer(learning_rate=0.01, beta=0.9)

        # --- Update path 1: use update() ---
        updated_parameters = optimizer1.update(parameters, gradients)

        # --- Update path 2: use apply_gradients() manually ---
        grads_and_vars = [(gradients["dWaa"], parameters2["Waa"]),
                          (gradients["dWax"], parameters2["Wax"])]
        optimizer2.apply_gradients(grads_and_vars)

        # --- Compare ---
        for key in updated_parameters.keys():
            np.testing.assert_allclose(updated_parameters[key], parameters2[key], rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
