import numpy as np


def compute_loss_and_grad(logits, y_true):
    """
    Computes the cross-entropy loss and its gradient w.r.t logits.

    Args:
        logits (ndarray): Shape (vocab_size, 1, T_x) or (vocab_size, T_x)
        y_true (list[int]): Ground truth indices

    Returns:
        loss (float)
        dy (ndarray): Gradient w.r.t. logits, same shape as logits
    """
    if logits.ndim == 2:
        logits = logits[:, np.newaxis, :]  # Ensure 3D shape: (vocab, 1, T_x)

    T_x = len(y_true)
    vocab_size = logits.shape[0]

    # Softmax (numerically stable)
    z = logits - np.max(logits, axis=0, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # Loss
    log_probs = np.log(probs + 1e-9)  # avoid log(0)
    loss = -sum(log_probs[y_true[t], 0, t] for t in range(T_x)) / T_x

    # Gradient
    dy = probs.copy()
    for t in range(T_x):
        dy[y_true[t], 0, t] -= 1
    dy /= T_x

    return loss, dy


def project_logit_grad_to_hidden(dy, Wy):
    """
    Projects the gradient of the loss w.r.t. output logits (dy) back to the hidden states (da).

    This is equivalent to: da_t = Wyᵀ · dy_t for each time step t.

    Args:
        dy (ndarray): Gradient of loss w.r.t. output logits, shape (n_y, 1, T_x)
        Wy (ndarray): Output layer weights, shape (n_y, n_a)

    Returns:
        da (ndarray): Gradient of loss w.r.t. hidden states, shape (n_a, 1, T_x)
    """
    n_y, _, T_x = dy.shape
    n_a = Wy.shape[1]

    da = np.zeros((n_a, 1, T_x))

    for t in range(T_x):
        da[:, :, t] = np.dot(Wy.T, dy[:, :, t])

    return da
