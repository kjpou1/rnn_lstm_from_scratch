import numpy as np


def compute_loss_and_grad(logits, y_true, reduction="mean"):
    """
    Compute the cross-entropy loss and its gradient w.r.t logits.

    Args:
        logits (ndarray): Raw output from the model. Shape:
                          - (vocab_size, T_x)
                          - OR (vocab_size, 1, T_x)
        y_true (list[int]): Ground truth token indices, length T_x
        reduction (str): "mean" or "sum". Whether to average the loss & gradient over time.

    Returns:
        loss (float): Total or average cross-entropy loss over time steps
        dy (ndarray): Gradient of the loss w.r.t logits, same shape as input logits
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Invalid reduction mode: {reduction}. Use 'mean' or 'sum'.")

    # --- Normalize shape ---
    if logits.ndim == 2:
        # Expand to (vocab_size, 1, T_x) for consistent indexing
        logits = logits[:, np.newaxis, :]

    vocab_size, _, T_x = logits.shape

    # --- Numerically stable softmax ---
    # Subtract max for numerical stability: logsumexp trick
    z = logits - np.max(logits, axis=0, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=0, keepdims=True)  # shape: (vocab, 1, T_x)

    # --- Cross-entropy loss ---
    # log_probs[y_true[t], 0, t] gives log(p_true) at each timestep
    log_probs = np.log(probs + 1e-9)  # avoid log(0)
    loss = -sum(log_probs[y_true[t], 0, t] for t in range(T_x))

    if reduction == "mean":
        loss /= T_x  # average over time steps

    # --- Gradient w.r.t. logits ---
    # ∂L/∂z = softmax - one_hot
    dy = probs.copy()
    for t in range(T_x):
        dy[y_true[t], 0, t] -= 1

    if reduction == "mean":
        dy /= T_x  # scale gradient like the loss

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
