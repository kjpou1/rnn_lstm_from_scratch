import numpy as np


def compute_output_layer_gradients(dy, a):
    """
    Compute gradients for output layer weights.

    Args:
        dy: ∂L/∂logits, shape (n_y, T_x)
        a: Dict or array of hidden states

    Returns:
        dict: dWya, dWy (if needed), dby
    """
    # print("dy.shape before squeeze:", dy.shape)
    dy = np.squeeze(dy, axis=1)
    # print("dy.shape:", dy.shape)
    n_y, T_x = dy.shape

    # Determine hidden size from any a[t]
    # Adapter to support both styles
    if isinstance(a, dict):
        n_a = a[0].shape[0]
        get_at = lambda t: a[t]  # shape (n_a, 1)
    else:
        if a.ndim == 3 and a.shape[1] == 1:
            a = np.squeeze(a, axis=1)  # (n_a, T_x)
        n_a = a.shape[0]
        get_at = lambda t: a[:, [t]]  # shape (n_a, 1)

    dWya = np.zeros((n_y, n_a))
    dby = np.zeros((n_y, 1))

    for t in range(T_x):
        a_t = get_at(t)  # works for both dict and array
        dWya += np.dot(dy[:, [t]], a_t.T)
        dby += dy[:, [t]]

    # TODO: consolidate dWy vs dWya naming across RNN/LSTM
    return {"dWy": dWya, "dWya": dWya, "dby": dby}
