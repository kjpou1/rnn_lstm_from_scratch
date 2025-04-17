import numpy as np

from activations.softmax import SoftmaxActivation
from activations.tanh import TanhActivation


def initialize_rnn_parameters(n_a, n_x, n_y, seed=1):
    """
    Initialize RNN parameters with small random values.

    Args:
        n_a (int): Number of hidden units (hidden size)
        n_x (int): Input size (vocabulary size)
        n_y (int): Output size (vocabulary size)
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing initialized parameters
    """
    np.random.seed(seed)
    return {
        "Wax": np.random.randn(n_a, n_x) * 0.01,  # Input to hidden
        "Waa": np.random.randn(n_a, n_a) * 0.01,  # Hidden to hidden
        "Wya": np.random.randn(n_y, n_a) * 0.01,  # Hidden to output
        "ba": np.zeros((n_a, 1)),  # Hidden bias
        "by": np.zeros((n_y, 1)),  # Output bias
    }


def update_parameters(parameters, gradients, lr):
    """
    Update RNN parameters using gradient descent.

    Args:
        parameters (dict): Current parameters
        gradients (dict): Gradients
        lr (float): Learning rate

    Returns:
        dict: Updated parameters
    """
    for param in parameters:
        parameters[param] -= lr * gradients["d" + param]
    return parameters


def rnn_cell_step(x_t, a_prev, parameters):
    """
    Single forward step of an RNN cell using modular activations.

    Args:
        x_t (ndarray): One-hot input vector at time t (vocab_size, 1)
        a_prev (ndarray): Previous hidden state (n_a, 1)
        parameters (dict): Dictionary of RNN weights and biases

    Returns:
        a_next (ndarray): Next hidden state (n_a, 1)
        y_pred (ndarray): Softmax prediction vector (n_y, 1)
        cache (tuple): Tuple of values needed for backpropagation
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

    # Prediction via softmax
    y_pred = SoftmaxActivation.forward(z_y)

    # Cache intermediate results for use in rnn_step_backward
    cache = (a_next, a_prev, x_t, parameters)

    return a_next, y_pred, z, cache


def rnn_forward(X, a0, parameters):
    """
    Forward pass through the full sequence.

    Args:
        X (list[int]): Input indices
        a0 (ndarray): Initial hidden state
        parameters (dict): Model parameters

    Returns:
        tuple: (y_hat, a, x)
    """
    vocab_size = parameters["Wax"].shape[1]
    a = {}
    y_hat = {}
    x = {}
    a[-1] = np.copy(a0)
    z_t = []

    for t, idx in enumerate(X):
        x_t = np.zeros((vocab_size, 1))
        if idx is not None:
            x_t[idx] = 1
        x[t] = x_t

        a[t], y_hat[t], z, _ = rnn_cell_step(x[t], a[t - 1], parameters)
        z_t.append(z)

    return y_hat, a, x, z_t


def rnn_step_backward(dy, da_next, cache):
    """
    Single backward step of an RNN cell.

    Args:
        dy (ndarray): Gradient of loss w.r.t. softmax output (n_y, 1)
        da_next (ndarray): Gradient of loss w.r.t. next hidden state (n_a, 1)
        cache (tuple): Values from forward pass (a_next, a_prev, x_t, z, parameters)

    Returns:
        grads (dict): Gradients for Wax, Waa, Wya, ba, by
        da_prev (ndarray): Gradient to be passed back to previous cell
    """
    (a_next, a_prev, x_t, z_t, parameters) = cache
    Waa, Wax, Wya = parameters["Waa"], parameters["Wax"], parameters["Wya"]

    # Gradients for output layer
    dWya = np.dot(dy, a_next.T)  # ∂L/∂Wya = dy · a_nextᵀ
    dby = dy  # ∂L/∂by = dy

    # Total gradient flowing into tanh from output and future timestep
    da = np.dot(Wya.T, dy) + da_next

    # Backprop through tanh activation
    dz = TanhActivation.backward(z_t) * da  # ∂L/∂z = ∂L/∂a * ∂a/∂z

    # Gradients for input and hidden weights
    dWax = np.dot(dz, x_t.T)  # ∂L/∂Wax = dz · x_tᵀ
    dWaa = np.dot(dz, a_prev.T)  # ∂L/∂Waa = dz · a_prevᵀ
    dba = dz  # ∂L/∂ba = dz

    # Gradient w.r.t. previous hidden state
    da_prev = np.dot(Waa.T, dz)  # ∂L/∂a_prev = Waaᵀ · dz

    grads = {
        "dWax": dWax,
        "dWaa": dWaa,
        "dWya": dWya,
        "dba": dba,
        "dby": dby,
    }

    return grads, da_prev


def rnn_backward(X, Y, parameters, cache):
    """
    Full backward pass through time.

    Args:
        X (list[int]): Input sequence
        Y (list[int]): Target sequence
        parameters (dict): Model parameters
        cache (tuple): Outputs from rnn_forward

    Returns:
        gradients (dict): Gradients w.r.t parameters
        a (dict): Hidden states
    """
    y_hat, a, x, z_t = cache
    vocab_size = parameters["Wax"].shape[1]
    n_a = parameters["Waa"].shape[0]

    # Initialize gradients
    gradients = {
        "dWax": np.zeros_like(parameters["Wax"]),
        "dWaa": np.zeros_like(parameters["Waa"]),
        "dWya": np.zeros_like(parameters["Wya"]),
        "dba": np.zeros_like(parameters["ba"]),
        "dby": np.zeros_like(parameters["by"]),
    }
    da_next = np.zeros((n_a, 1))

    # Loop backward through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1  # Derivative of cross-entropy loss w.r.t softmax output

        cache_t = (a[t], a[t - 1], x[t], z_t[t], parameters)
        step_grads, da_next = rnn_step_backward(dy, da_next, cache_t)

        for key in gradients.keys():
            gradients[key] += step_grads[key]

    return gradients, a
