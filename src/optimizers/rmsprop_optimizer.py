import numpy as np

from .base_optimizer import BaseOptimizer


class RMSPropOptimizer(BaseOptimizer):
    """
    RMSProp Optimizer.

    RMSProp adapts the learning rate for each parameter individually by maintaining
    a moving average of the squared gradients.

    Update rules:

        s := β * s + (1 - β) * (∇J(θ))²        # Update running average
        θ := θ - α * ∇J(θ) / (sqrt(s) + ε)     # Parameter update

    Where:
        - s : running average of squared gradients
        - β : decay factor (commonly 0.9)
        - α : learning rate
        - ε : small value to avoid division by zero
    """

    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        """
        Initialize the RMSProp optimizer.

        Args:
            learning_rate (float): Step size α.
            beta (float): Decay rate β for moving average.
            epsilon (float): Small value to prevent division by zero.
        """
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}  # Tracks running average for each parameter

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients using RMSProp update rule.

        Args:
            grads_and_vars (list): List of (gradient, parameter) tuples.
        """
        for grad, param in grads_and_vars:
            param_id = id(param)

            # Initialize moving average if first time seeing this param
            if param_id not in self.s:
                self.s[param_id] = np.zeros_like(grad)

            # Update running average: s = β * s + (1 - β) * grad²
            self.s[param_id] = self.beta * self.s[param_id] + (1 - self.beta) * (
                grad**2
            )

            # Update param: θ := θ - α * grad / (sqrt(s) + ε)
            param -= (
                self.learning_rate * grad / (np.sqrt(self.s[param_id]) + self.epsilon)
            )
