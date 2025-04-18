# src/activations/softmax.py

import numpy as np

from .base_activation import BaseActivation


class SoftmaxActivation(BaseActivation):
    """
    Softmax Activation.

    Forward:
        Converts raw logits into a probability distribution across classes.
        Applies element-wise exponential followed by normalization.

    Backward:
        Computes the Jacobian-vector product of the softmax function.
        Efficiently supports dL/dz given dL/da without computing full Jacobian.

    Note:
    - This class assumes that softmax is applied to **vectors**, not batches.
    - Backprop with softmax is rarely used directly unless doing custom losses.
    """

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax output.

        Args:
            x (ndarray): Input logits (n, 1) or (n,)

        Returns:
            ndarray: Probability vector (same shape as x)
        """
        x = x - np.max(x)  # for numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    @staticmethod
    def backward(s: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix of the softmax output.

        Args:
            s (ndarray): Softmax probabilities (output from forward), shape (n,) or (n,1)

        Returns:
            ndarray: Jacobian matrix of shape (n, n)

        Note:
            The Jacobian J of the softmax function is:

                J_ij = s_i * (δ_ij - s_j)

            For most classification tasks, you can skip this and use:

                ∇L = y_pred - y_true

            when using softmax with cross-entropy loss.
        """
        s = s.reshape(-1, 1)  # Ensure column vector
        return np.diagflat(s) - np.dot(s, s.T)
