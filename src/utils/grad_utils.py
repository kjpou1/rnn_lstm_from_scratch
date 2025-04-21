import numpy as np


def compute_output_layer_gradients(dy, a):
    """
    Computes gradients for the output layer weights and biases
    by projecting the loss gradient (dy) back through the output layer.

    Args:
        dy (ndarray): Gradient of loss w.r.t. logits,
                      shape (n_y, m, T_x)
        a  (ndarray): Hidden states from forward pass,
                      shape (n_a, m, T_x)

    Returns:
        dict: Dictionary with:
            - "dWy": Gradient w.r.t. output weights, shape (n_y, n_a)
            - "dby": Gradient w.r.t. output biases, shape (n_y, 1)
    """
    n_y, _, T_x = dy.shape
    n_a, _, _ = a.shape

    dy_flat = dy.reshape(n_y, -1)  # (n_y, m*T_x)
    a_flat = a.reshape(n_a, -1)  # (n_a, m*T_x)

    dWy = np.dot(dy_flat, a_flat.T)  # (n_y, n_a)
    dby = np.sum(dy, axis=(1, 2)).reshape(-1, 1)  # (n_y, 1)

    return {"dWy": dWy, "dby": dby}
