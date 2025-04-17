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
    def backward(x: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix of softmax at x.

        Args:
            x (ndarray): Same input as used in forward(x)

        Returns:
            ndarray: Jacobian matrix of softmax (n, n)

        Note:
            Only needed if doing full Jacobian backprop.
            In practice, most use dL/dz = y_pred - y_true for softmax + cross-entropy.
        """
        s = SoftmaxActivation.forward(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
