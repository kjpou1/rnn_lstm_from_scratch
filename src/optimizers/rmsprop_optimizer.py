import numpy as np
from .optimizer import Optimizer

class RMSPropOptimizer(Optimizer):
    """
    RMSProp Optimizer.

    RMSProp adapts the learning rate for each parameter individually by maintaining
    a moving average of the squared gradients.

    Update rules:

        s = β * s + (1 - β) * (grad ** 2)      # Update running average
        θ = θ - α * grad / (sqrt(s) + ε)       # Update parameter

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
            beta (float): Decay rate β for running average.
            epsilon (float): Small number to prevent division by zero.
        """
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}  # Dictionary to store the running average (s) per parameter

    def update(self, parameters, gradients):
        """
        Convenience function for scratch RNN training loops.

        Constructs grads_and_vars list and calls apply_gradients internally.

        Args:
            parameters (dict): Dictionary of parameters.
            gradients (dict): Dictionary of gradients.

        Returns:
            dict: Updated parameters.
        """
        grads_and_vars = [(gradients["d" + key], parameters[key]) for key in parameters.keys()]
        self.apply_gradients(grads_and_vars)
        return parameters

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to parameters using RMSProp update rule.

        Args:
            grads_and_vars (list): List of (gradient, parameter) tuples.
        """
        for grad, param in grads_and_vars:
            param_id = id(param)

            # Initialize running average if first time seeing this parameter
            if param_id not in self.s:
                self.s[param_id] = np.zeros_like(grad)

            # --- Update running average of squared gradients ---
            # s := β * s + (1 - β) * (grad ** 2)
            self.s[param_id] = self.beta * self.s[param_id] + (1 - self.beta) * (grad ** 2)

            # --- Update parameter ---
            # θ := θ - α * grad / (sqrt(s) + ε)
            param -= self.learning_rate * grad / (np.sqrt(self.s[param_id]) + self.epsilon)

