# src/activations/sigmoid.py

import numpy as np

from .base_activation import BaseActivation


class SigmoidActivation(BaseActivation):
    """
    Sigmoid Activation Function.

    Maps input values to the (0, 1) range:
        a = 1 / (1 + exp(-x))

    Useful in binary classification or gating mechanisms (e.g., LSTM gates).

    Derivative:
        da/dx = a * (1 - a)
    """

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function element-wise.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Activated output.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of sigmoid w.r.t. its input x.

        Args:
            x (np.ndarray): Pre-activation input (same as passed to forward)

        Returns:
            np.ndarray: Derivative of sigmoid.
        """
        s = SigmoidActivation.forward(x)
        return s * (1 - s)
