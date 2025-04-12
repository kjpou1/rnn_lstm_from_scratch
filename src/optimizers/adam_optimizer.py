import numpy as np
from .optimizer import Optimizer

class AdamOptimizer(Optimizer):
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
        self.t = 0    # Time step counter (for bias correction)

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to parameters using Adam optimization rule.

        Args:
            grads_and_vars (list): List of (gradient, parameter) tuples.
        """
        self.t += 1  # Increment timestep

        for grad, param in grads_and_vars:
            param_id = id(param)

            # Initialize if first time seeing this parameter
            if param_id not in self.ms:
                self.ms[param_id] = np.zeros_like(param)
                self.vs[param_id] = np.zeros_like(param)

            m = self.ms[param_id]
            v = self.vs[param_id]

            # Update biased first moment estimate (Momentum)
            m = self.beta1 * m + (1 - self.beta1) * grad

            # Update biased second raw moment estimate (RMSProp)
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment
            m_hat = m / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second moment
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameter
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Store updates
            self.ms[param_id] = m
            self.vs[param_id] = v

    def update(self, parameters, gradients):
        """
        Scratch-style parameter update for RNN training loops.

        Args:
            parameters (dict): Parameters dictionary {Waa, Wax, Wya, ba, by}.
            gradients (dict): Gradients dictionary {dWaa, dWax, dWya, dba, dby}.

        Returns:
            dict: Updated parameters.
        """
        grads_and_vars = [(gradients["d" + key], parameters[key]) for key in parameters.keys()]
        self.apply_gradients(grads_and_vars)
        return parameters
