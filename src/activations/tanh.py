# src/activations/tanh.py

import numpy as np

from .base_activation import BaseActivation


class TanhActivation(BaseActivation):
    """
    Hyperbolic Tangent Activation Function (tanh).

    Forward:
        a = tanh(x)

    Backward:
        da/dx = 1 - tanh(x)^2
    """

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Compute tanh activation.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: tanh(x)
        """
        return np.tanh(x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Compute derivative of tanh activation with respect to x.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Derivative (1 - tanh(x)^2)
        """
        return 1 - np.tanh(x) ** 2
