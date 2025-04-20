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
        Compute softmax probabilities from raw logits with numerical stability.

        Args:
            x (ndarray): Input array of logits.
                - Shape (n_classes, 1) for single example
                - Shape (n_classes,) is also supported

        Returns:
            ndarray: Softmax probabilities (same shape as input)
        """
        # 🧠 Numerical Stability Trick:
        # Subtract the max value to avoid large exponents (e.g. exp(1000))
        # This ensures the result of exp(x - max) is safe from overflow
        x_max = np.max(x, axis=0, keepdims=True)  # shape: (1, 1) or (1,)

        # Compute exponentials after stabilization
        exp_x = np.exp(x - x_max)

        # Normalize across the class axis (column vector case → axis=0)
        softmax_output = exp_x / np.sum(exp_x, axis=0, keepdims=True)

        return softmax_output

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
