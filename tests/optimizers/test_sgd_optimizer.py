import unittest
import numpy as np
from src.optimizers.sgd_optimizer import SGDOptimizer

class TestSGDOptimizer(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "Waa": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "Wax": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        self.gradients = {
            "dWaa": np.array([[0.1, 0.1], [0.1, 0.1]]),
            "dWax": np.array([[0.2, 0.2], [0.2, 0.2]]),
        }
        self.learning_rate = 0.5

    def test_update_matches_manual_formula(self):
        optimizer = SGDOptimizer(learning_rate=self.learning_rate)

        # Clone to avoid in-place mutation
        parameters_copy = {k: np.copy(v) for k, v in self.parameters.items()}
        updated_parameters = optimizer.update(parameters_copy, self.gradients)

        expected_Waa = self.parameters["Waa"] - self.learning_rate * self.gradients["dWaa"]
        expected_Wax = self.parameters["Wax"] - self.learning_rate * self.gradients["dWax"]

        np.testing.assert_allclose(updated_parameters["Waa"], expected_Waa, rtol=1e-6)
        np.testing.assert_allclose(updated_parameters["Wax"], expected_Wax, rtol=1e-6)

    def test_update_vs_apply_gradients(self):
        optimizer1 = SGDOptimizer(learning_rate=self.learning_rate)
        optimizer2 = SGDOptimizer(learning_rate=self.learning_rate)

        params1 = {k: np.copy(v) for k, v in self.parameters.items()}
        params2 = {k: np.copy(v) for k, v in self.parameters.items()}

        updated_params = optimizer1.update(params1, self.gradients)

        grads_and_vars = [
            (self.gradients["dWaa"], params2["Waa"]),
            (self.gradients["dWax"], params2["Wax"]),
        ]
        optimizer2.apply_gradients(grads_and_vars)

        for key in updated_params.keys():
            np.testing.assert_allclose(updated_params[key], params2[key], rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
