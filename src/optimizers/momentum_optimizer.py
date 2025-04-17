# src/optimizers/momentum_optimizer.py

import numpy as np

from .base_optimizer import BaseOptimizer


class MomentumOptimizer(BaseOptimizer):
    """
    Momentum Optimizer.

    Extends basic SGD by adding a velocity term to each parameter update:

        v := β * v - α * ∇J(θ)
        θ := θ + v

    This helps accelerate convergence by smoothing updates and reducing oscillations,
    especially in ravines or along steep gradients.

    Where:
        - v: velocity vector (accumulated direction of past gradients)
        - β (beta): momentum coefficient (e.g., 0.9) – controls how much of the previous velocity to retain
        - α (alpha): learning rate – scales the magnitude of the gradient step
        - ∇J(θ): gradient of the loss function with respect to parameter θ
        - θ: model parameter being updated

    Attributes:
        learning_rate (float): Step size for updates.
        beta (float): Coefficient that scales the contribution of the previous update.
        velocities (dict): Stores parameter-specific velocity buffers.
    """

    def __init__(self, learning_rate=0.01, beta=0.9):
        """
        Initialize the Momentum optimizer.

        Args:
            learning_rate (float): Step size used to scale gradients.
            beta (float): Smoothing factor for velocity accumulation.
        """
        super().__init__(learning_rate)
        self.beta = beta
        self.velocities = {}  # Keyed by parameter ID to persist update history

    def apply_gradients(self, grads_and_vars):
        """
        Applies momentum-based updates to parameters.

        Args:
            grads_and_vars (list of tuples): Each tuple contains:
                - grad (ndarray): The gradient of the loss w.r.t. a parameter.
                - param (ndarray): The parameter to update.
        """
        for grad, param in grads_and_vars:
            param_id = id(param)

            # Initialize velocity buffer if needed
            if param_id not in self.velocities:
                self.velocities[param_id] = np.zeros_like(param)

            v_prev = self.velocities[param_id]

            # ---- Momentum update rule ----
            # v := β * v - α * grad
            v_new = self.beta * v_prev - self.learning_rate * grad

            # Save the new velocity
            self.velocities[param_id] = v_new

            # θ := θ + v
            param += v_new
