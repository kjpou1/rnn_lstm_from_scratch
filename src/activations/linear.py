# src/activations/linear.py

import numpy as np

from .base_activation import BaseActivation


class LinearActivation(BaseActivation):
    """
    Linear Activation Function (Identity).

    Used when no non-linearity is applied:
        a = x

    Derivative:
        da/dx = 1
    """

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Return input directly (identity function).

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Same as input.
        """
        return x

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Return derivative of identity function, which is 1.

        Args:
            x (np.ndarray): Pre-activation input (ignored)

        Returns:
            np.ndarray: Ones with the same shape as x.
        """
        return np.ones_like(x)
