import numpy as np

from .activations.softmax import SoftmaxActivation
from .activations.tanh import TanhActivation


def initialize_rnn_parameters(n_a, n_x, n_y, seed=1):
    """
    Initialize RNN weights using small random values and zero biases.

    Args:
        n_a: Number of hidden units
        n_x: Input size (vocab size)
        n_y: Output size (same as vocab size for character-level RNN)
        seed: Random seed for reproducibility

    Returns:
        dict: Parameters {Wax, Waa, Wya, ba, by}
    """
    np.random.seed(seed)
    return {
        "Wax": np.random.randn(n_a, n_x) * 0.01,  # input weight matrix
        "Waa": np.random.randn(n_a, n_a) * 0.01,  # hidden-to-hidden weight matrix
        "Wya": np.random.randn(n_y, n_a) * 0.01,  # hidden-to-output weight matrix
        "ba": np.zeros((n_a, 1)),  # hidden bias
        "by": np.zeros((n_y, 1)),  # output bias
    }


def update_parameters(parameters, gradients, lr):
    """
    Simple SGD update.

    Args:
        parameters: Current model parameters
        gradients: Computed gradients
        lr: Learning rate

    Returns:
        Updated parameters
    """
    for param in parameters:
        parameters[param] -= lr * gradients["d" + param]
    return parameters


def rnn_cell_step(x_t, a_prev, parameters):
    """
    Single RNN cell forward pass.

    Args:
        x_t: One-hot vector at timestep t, shape (n_x, 1)
        a_prev: Hidden state at t-1, shape (n_a, 1)
        parameters: Dictionary of weights

    Returns:
        a_next: Hidden state at t
        y_pred: Softmax probabilities (output)
        z_y: Raw output logits
        cache: Tuple for backward pass
    """
    Waa, Wax, Wya, ba, by = (
        parameters["Waa"],
        parameters["Wax"],
        parameters["Wya"],
        parameters["ba"],
        parameters["by"],
    )

    # Linear combination before activation (used in both forward and backward passes)
    z = np.dot(Wax, x_t) + np.dot(Waa, a_prev) + ba

    # Hidden state activation (tanh)
    a_next = TanhActivation.forward(z)

    # Output logits
    z_y = np.dot(Wya, a_next) + by

    # Cache intermediate results for use in rnn_step_backward
    cache = (a_next, a_prev, x_t, parameters)

    return a_next, z_y, z, cache


def rnn_forward(X, a0, parameters):
    """
    Forward pass over the full sequence.

    Args:
        X: List of token indices (length T_x)
        a0: Initial hidden state, shape (n_a, 1)
        parameters: Dictionary of model weights

    Returns:
        y_hat: Dict of predictions at each t
        a: Dict of hidden states
        x: Dict of one-hot vectors
        z_t: Array of logits across time (n_y, T_x)
    """
    vocab_size = parameters["Wax"].shape[1]
    a = {}
    x = {}
    logits_list = []
    z_t = []
    a[-1] = np.copy(a0)

    for t, idx in enumerate(X):
        x_t = np.zeros((vocab_size, 1))
        if idx is not None:
            x_t[idx] = 1
        x[t] = x_t

        a[t], z_y, z, _ = rnn_cell_step(x[t], a[t - 1], parameters)
        logits_list.append(z_y)  # z_y: (n_y, 1)
        z_t.append(z)

    logits = np.concatenate(logits_list, axis=1)  # z_t: (n_y, T_x)

    return a, x, logits, z_t


def rnn_step_backward(dy, da_next, cache):
    """
    Backprop for a single RNN cell.

    Args:
        dy: Gradient of loss w.r.t softmax output, shape (n_y, 1)
        da_next: Gradient w.r.t next hidden state (from future timestep)
        cache: Tuple from forward pass

    Returns:
        grads: Dictionary of gradients
        da_prev: Gradient to propagate back to t-1
    """
    a_next, a_prev, x_t, z_t, parameters = cache
    Waa, Wax, Wya = parameters["Waa"], parameters["Wax"], parameters["Wya"]

    # ∂L/∂Wya = dy · a_nextᵀ
    dWya = np.dot(dy, a_next.T)

    # ∂L/∂by = dy
    dby = dy

    # ∂L/∂a = Wyaᵀ · dy + da_next
    da = np.dot(Wya.T, dy) + da_next

    # Backprop through tanh activation
    dz = TanhActivation.backward(z_t) * da  # ∂L/∂z = ∂L/∂a * ∂a/∂z  ← tanh derivative

    # ∂L/∂Wax = dz · x_tᵀ
    dWax = np.dot(dz, x_t.T)

    # ∂L/∂Waa = dz · a_prevᵀ
    dWaa = np.dot(dz, a_prev.T)

    # ∂L/∂ba = dz
    dba = dz

    # ∂L/∂a_prev = Waaᵀ · dz
    da_prev = np.dot(Waa.T, dz)

    return {
        "dWax": dWax,
        "dWaa": dWaa,
        "dWya": dWya,
        "dba": dba,
        "dby": dby,
    }, da_prev


def rnn_backward(X, Y, parameters, cache):
    """
    Full backward pass through time (BPTT).

    Args:
        X: List of input indices (T_x)
        Y: List of target indices (T_x)
        parameters: Model weights
        cache: Outputs from rnn_forward

    Returns:
        gradients: Dictionary of dWax, dWaa, dWya, dba, dby
        a: Hidden states over time
    """
    a, x, logits, z_t = cache
    n_a = parameters["Waa"].shape[0]

    gradients = {
        "dWax": np.zeros_like(parameters["Wax"]),
        "dWaa": np.zeros_like(parameters["Waa"]),
        "dWya": np.zeros_like(parameters["Wya"]),
        "dba": np.zeros_like(parameters["ba"]),
        "dby": np.zeros_like(parameters["by"]),
    }
    da_next = np.zeros((n_a, 1))

    # 🔁 Precompute softmax across time for all logits (n_y, T_x)
    y_hat = SoftmaxActivation.forward(logits)

    for t in reversed(range(len(X))):
        dy = y_hat[:, [t]]  # shape: (n_y, 1)

        # 🧮 Cross-entropy gradient: ∂L/∂z = y_hat - y_one_hot
        # Equivalent to subtracting 1 from the predicted prob at the true class index
        dy[Y[t], 0] -= 1

        cache_t = (a[t], a[t - 1], x[t], z_t[t], parameters)
        step_grads, da_next = rnn_step_backward(dy, da_next, cache_t)

        for key in gradients:
            gradients[key] += step_grads[key]

    return gradients, a
