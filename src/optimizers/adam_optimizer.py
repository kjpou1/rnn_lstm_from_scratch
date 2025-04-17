# src/optimizers/adam_optimizer.py

import numpy as np

from .base_optimizer import BaseOptimizer


class AdamOptimizer(BaseOptimizer):
    """
    Adam Optimizer.

    Adam = Adaptive Moment Estimation.
    It combines:
    - Momentum (1st moment / mean of gradients)
    - RMSProp (2nd moment / uncentered variance of gradients)

    Update Equations:

    1. Momentum update (First moment estimate):
        mₜ = β₁ * mₜ₋₁ + (1 - β₁) * gₜ

    2. RMSProp update (Second moment estimate):
        vₜ = β₂ * vₜ₋₁ + (1 - β₂) * (gₜ)²

    3. Bias Correction:
        m̂ₜ = mₜ / (1 - β₁ᵗ)
        v̂ₜ = vₜ / (1 - β₂ᵗ)

    4. Parameter update:
        θ = θ - α * m̂ₜ / (sqrt(v̂ₜ) + ε)

    Where:
    - α: Learning rate
    - β₁: Decay rate for first moment (typically 0.9)
    - β₂: Decay rate for second moment (typically 0.999)
    - ε: Small constant for numerical stability (e.g., 1e-8)
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.ms = {}  # First moment estimates (mₜ)
        self.vs = {}  # Second moment estimates (vₜ)
        self.t = 0  # Time step counter (used for bias correction)

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to parameters using Adam optimization rule.

        Args:
            grads_and_vars (list): List of (gradient, parameter) tuples.
        """
        self.t += 1  # ⏱ Increment timestep

        for grad, param in grads_and_vars:
            param_id = id(param)

            # Initialize moment estimates if not present
            if param_id not in self.ms:
                self.ms[param_id] = np.zeros_like(param)
                self.vs[param_id] = np.zeros_like(param)

            m = self.ms[param_id]
            v = self.vs[param_id]

            # --- Momentum update (first moment) ---
            # mₜ = β₁ * mₜ₋₁ + (1 - β₁) * gₜ
            m = self.beta1 * m + (1 - self.beta1) * grad

            # --- RMSProp update (second moment) ---
            # vₜ = β₂ * vₜ₋₁ + (1 - β₂) * (gₜ)²
            v = self.beta2 * v + (1 - self.beta2) * (grad**2)

            # --- Bias-corrected moments ---
            # m̂ₜ = m / (1 - β₁ᵗ)
            m_hat = m / (1 - self.beta1**self.t)
            # v̂ₜ = v / (1 - β₂ᵗ)
            v_hat = v / (1 - self.beta2**self.t)

            # --- Parameter update ---
            # θ = θ - α * m̂ₜ / (sqrt(v̂ₜ) + ε)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Save updated moments
            self.ms[param_id] = m
            self.vs[param_id] = v
