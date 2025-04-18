# src/activations/sigmoid.py

import numpy as np

from .base_activation import BaseActivation


class SigmoidActivation(BaseActivation):
    """
    Sigmoid Activation Function.

    Maps input values to the (0, 1) range:
        a = 1 / (1 + exp(-x))

    Commonly used for binary classification and gating (e.g., LSTM).

    Derivative:
        da/dx = a * (1 - a), where a is the sigmoid output
    """

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function element-wise.

        Args:
            x (np.ndarray): Raw input (pre-activation).

        Returns:
            np.ndarray: Activated output (sigmoid(x)).
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of sigmoid w.r.t. its output.

        Args:
            x (np.ndarray): Sigmoid output (a = sigmoid(z)).

        Returns:
            np.ndarray: Element-wise derivative a * (1 - a).
        """
        return x * (1 - x)
