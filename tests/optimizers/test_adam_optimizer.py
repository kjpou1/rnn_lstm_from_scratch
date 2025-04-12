import unittest
import numpy as np
from src.optimizers.adam_optimizer import AdamOptimizer

class TestAdamOptimizer(unittest.TestCase):
    def test_update_matches_manual_formula(self):
        """
        Test that optimizer.update() matches manual Adam math after 1 step.
        """

        # Fake parameters and gradients
        Waa = np.array([[1.0, 2.0], [3.0, 4.0]])
        Wax = np.array([[5.0, 6.0], [7.0, 8.0]])
        parameters = {"Waa": np.copy(Waa), "Wax": np.copy(Wax)}

        gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }

        # Hyperparameters
        lr = 0.001       # Learning rate (α)
        beta1 = 0.9      # Exponential decay for first moment estimate
        beta2 = 0.999    # Exponential decay for second moment estimate
        epsilon = 1e-8   # Small value to avoid division by zero

        optimizer = AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

        # ---- Perform optimizer update ----
        updated_parameters = optimizer.update(parameters, gradients)

        # ---- Manual Calculation following Adam formulas ----
        
        # 1. First moment estimate (m):
        # m = β₁ * m_prev + (1 - β₁) * gradient
        m_dWaa = (1 - beta1) * gradients["dWaa"]
        m_dWax = (1 - beta1) * gradients["dWax"]

        # 2. Second moment estimate (v):
        # v = β₂ * v_prev + (1 - β₂) * (gradient)^2
        v_dWaa = (1 - beta2) * (gradients["dWaa"] ** 2)
        v_dWax = (1 - beta2) * (gradients["dWax"] ** 2)

        # 3. Bias correction:
        # m̂ = m / (1 - β₁ᵗ)
        # v̂ = v / (1 - β₂ᵗ)
        m_hat_dWaa = m_dWaa / (1 - beta1)
        m_hat_dWax = m_dWax / (1 - beta1)

        v_hat_dWaa = v_dWaa / (1 - beta2)
        v_hat_dWax = v_dWax / (1 - beta2)

        # 4. Update rule:
        # θ = θ - α * m̂ / (sqrt(v̂) + ε)
        expected_Waa = Waa - lr * m_hat_dWaa / (np.sqrt(v_hat_dWaa) + epsilon)
        expected_Wax = Wax - lr * m_hat_dWax / (np.sqrt(v_hat_dWax) + epsilon)

        # ---- Assertions ----
        np.testing.assert_allclose(updated_parameters["Waa"], expected_Waa, rtol=1e-6)
        np.testing.assert_allclose(updated_parameters["Wax"], expected_Wax, rtol=1e-6)

    def test_adam_update_vs_apply_gradients(self):
        """
        Test that optimizer.update() and optimizer.apply_gradients() produce the same result.
        """

        # Setup parameters and gradients
        parameters = {
            "Waa": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Wax": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }

        parameters2 = {k: np.copy(v) for k, v in parameters.items()}

        optimizer1 = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        optimizer2 = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

        # Path 1: update
        updated_parameters = optimizer1.update(parameters, gradients)

        # Path 2: manual apply_gradients
        grads_and_vars = [
            (gradients["dWaa"], parameters2["Waa"]),
            (gradients["dWax"], parameters2["Wax"]),
        ]
        optimizer2.apply_gradients(grads_and_vars)

        for key in updated_parameters.keys():
            np.testing.assert_allclose(updated_parameters[key], parameters2[key], rtol=1e-6)

    def test_parameter_mismatch_should_fail(self):
        """
        Test that using wrong hyperparameters should result in different parameter updates.
        """

        parameters = {
            "Waa": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Wax": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        gradients = {
            "dWaa": np.array([[0.5, 0.5], [0.5, 0.5]]),
            "dWax": np.array([[1.0, 1.0], [1.0, 1.0]]),
        }

        parameters2 = {k: np.copy(v) for k, v in parameters.items()}

        optimizer_correct = AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
        optimizer_wrong = AdamOptimizer(learning_rate=0.005, beta1=0.85, beta2=0.95)

        updated_parameters = optimizer_correct.update(parameters, gradients)
        wrong_updated_parameters = optimizer_wrong.update(parameters2, gradients)

        # Assert they are NOT close
        for key in updated_parameters.keys():
            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(updated_parameters[key], wrong_updated_parameters[key], rtol=1e-6)

    def test_manual_calculation_vs_wrong_optimizer_should_fail(self):
        """
        Test that a manual correct calculation and a wrong optimizer update do NOT match.

        This ensures that manual math validates the optimizer behavior independently.
        """

        # --- Setup parameters and gradients ---
        Waa = np.array([[1.0, 2.0], [3.0, 4.0]])
        Wax = np.array([[5.0, 6.0], [7.0, 8.0]])
        parameters = {"Waa": np.copy(Waa), "Wax": np.copy(Wax)}

        gradients = {
            "dWaa": np.array([[0.5, 0.5], [0.5, 0.5]]),
            "dWax": np.array([[1.0, 1.0], [1.0, 1.0]]),
        }

        # --- Correct hyperparameters ---
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        optimizer = AdamOptimizer(learning_rate=0.005, beta1=0.85, beta2=0.95, epsilon=epsilon)  # Wrong values

        # --- Optimizer update ---
        updated_parameters = optimizer.update(parameters, gradients)

        # --- Manual Correct Calculation (Adam math) ---

        # 1. First moment estimate (mₜ = β₁ * mₜ₋₁ + (1 - β₁) * grad)
        m_dWaa = (1 - 0.9) * gradients["dWaa"]
        m_dWax = (1 - 0.9) * gradients["dWax"]

        # 2. Second moment estimate (vₜ = β₂ * vₜ₋₁ + (1 - β₂) * grad²)
        v_dWaa = (1 - 0.999) * (gradients["dWaa"] ** 2)
        v_dWax = (1 - 0.999) * (gradients["dWax"] ** 2)

        # 3. Bias correction
        m_hat_dWaa = m_dWaa / (1 - 0.9)
        m_hat_dWax = m_dWax / (1 - 0.9)

        v_hat_dWaa = v_dWaa / (1 - 0.999)
        v_hat_dWax = v_dWax / (1 - 0.999)

        # 4. Manual expected parameter update
        expected_Waa = Waa - 0.001 * m_hat_dWaa / (np.sqrt(v_hat_dWaa) + epsilon)
        expected_Wax = Wax - 0.001 * m_hat_dWax / (np.sqrt(v_hat_dWax) + epsilon)

        # --- Assertions (force fail) ---
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(updated_parameters["Waa"], expected_Waa, rtol=1e-6)

        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(updated_parameters["Wax"], expected_Wax, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
