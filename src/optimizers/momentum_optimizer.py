# src/optimizers/momentum_optimizer.py

from .optimizer import Optimizer
import numpy as np

class MomentumOptimizer(Optimizer):
    """
    Momentum Optimizer.

    Extends basic SGD by adding a velocity term to smooth updates:

        v := β * v - α * ∇J(θ)
        θ := θ + v

    Where:
    - β is the momentum coefficient (typically 0.9)
    - α is the learning rate
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize the Momentum optimizer.

        Args:
            learning_rate (float): Step size for updates.
            momentum (float): Momentum factor (0.9 by default).
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}  # Stores velocity tensors

    def update(self, parameters, gradients):
        """
        Traditional scratch RNN update.

        Args:
            parameters (dict): Model parameters.
            gradients (dict): Gradients.
            
        Returns:
            Updated parameters dict.
        """
        grads_and_vars = [(gradients["d" + key], parameters[key]) for key in parameters.keys()]
        self.apply_gradients(grads_and_vars)
        return parameters

    def apply_gradients(self, grads_and_vars):
        """
        Apply parameter updates using momentum.

        Args:
            grads_and_vars (list): List of (gradient, parameter) tuples.
        """
        for grad, param in grads_and_vars:
            param_id = id(param)  # Unique key for each parameter

            if param_id not in self.velocities:
                self.velocities[param_id] = np.zeros_like(param)

            # Update velocity
            v = self.velocities[param_id]
            v = self.momentum * v - self.learning_rate * grad

            # Save updated velocity
            self.velocities[param_id] = v

            # Update parameter
            param += v
