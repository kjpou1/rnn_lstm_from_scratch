import numpy as np

from ..activations.softmax import SoftmaxActivation
from ..activations.tanh import TanhActivation


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


def rnn_step_backward(da, cache):
    """
    Backward pass for a single RNN cell (recurrent path only — excludes output layer).

    This computes the gradients with respect to the input weights (Wax),
    recurrent weights (Waa), bias (ba), and previous hidden state (a_prev),
    given the upstream gradient ∂L/∂a from the current timestep.

    Args:
        da (ndarray): Gradient of loss w.r.t. hidden state at current timestep, shape (n_a, 1)
        cache (tuple): Values from forward pass: (a_next, a_prev, x_t, z_t, parameters)

    Returns:
        grads (dict): Gradients w.r.t. parameters — keys: dWax, dWaa, dba
        da_prev (ndarray): Gradient w.r.t. previous hidden state, shape (n_a, 1)
    """
    a_next, a_prev, x_t, z_t, parameters = cache
    Waa, Wax = parameters["Waa"], parameters["Wax"]

    # Backprop through tanh activation
    # ∂L/∂z = ∂L/∂a * (1 - tanh²(z))
    dz = TanhActivation.backward(z_t) * da

    # Gradients w.r.t. parameters
    # ∂L/∂Wax = dz · x_tᵀ
    dWax = np.dot(dz, x_t.T)

    # ∂L/∂Waa = dz · a_prevᵀ
    dWaa = np.dot(dz, a_prev.T)

    # ∂L/∂ba = dz
    dba = dz

    # Gradient w.r.t. previous hidden state
    # ∂L/∂a_prev = Waaᵀ · dz
    da_prev = np.dot(Waa.T, dz)

    return {"dWax": dWax, "dWaa": dWaa, "dba": dba}, da_prev


def rnn_backward(da, parameters, cache):
    """
    Backward pass through the entire sequence (BPTT) using ∂L/∂a from logits.

    This function accumulates parameter gradients over all timesteps in the sequence
    by recursively applying `rnn_step_backward`.

    Args:
        da (ndarray): Gradient of the loss w.r.t hidden states, shape (n_a, 1, T_x)
        parameters (dict): RNN weights: Wax, Waa, ba
        cache (tuple): Outputs from rnn_forward → (a, x, logits, z_t)

    Returns:
        gradients (dict): Accumulated gradients {dWax, dWaa, dba}
        a (dict): Hidden states over time, indexed by timestep
    """
    a, x, _, z_t = cache
    n_a = parameters["Waa"].shape[0]
    T_x = len(a) - 1  # because a[-1] is the initial hidden state

    # Initialize gradients
    gradients = {
        "dWax": np.zeros_like(parameters["Wax"]),
        "dWaa": np.zeros_like(parameters["Waa"]),
        "dba": np.zeros_like(parameters["ba"]),
    }

    da_next = np.zeros((n_a, 1))  # ∂L/∂a for the next timestep (starts at 0)

    # Backpropagate through time
    for t in reversed(range(T_x)):
        # Slice ∂L/∂a for current timestep and add running da_next
        da_t = da[:, 0, [t]]  # shape (n_a, 1)
        da_total = da_t + da_next  # total gradient flowing into this cell

        # Retrieve cache for timestep t and perform backward step
        cache_t = (a[t], a[t - 1], x[t], z_t[t], parameters)
        step_grads, da_next = rnn_step_backward(da_total, cache_t)

        # Accumulate gradients over time
        gradients["dWax"] += step_grads["dWax"]
        gradients["dWaa"] += step_grads["dWaa"]
        gradients["dba"] += step_grads["dba"]

    return gradients, a
