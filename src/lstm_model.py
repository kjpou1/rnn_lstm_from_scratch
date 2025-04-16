import numpy as np

from src.utils import sigmoid, softmax


def initialize_lstm_parameters(n_a, n_x, n_y, seed=1):
    np.random.seed(seed)

    def rand(shape):
        return np.random.randn(*shape) * 0.01

    return {
        "Wf": rand((n_a, n_a + n_x)),
        "bf": np.zeros((n_a, 1)),
        "Wi": rand((n_a, n_a + n_x)),
        "bi": np.zeros((n_a, 1)),
        "Wc": rand((n_a, n_a + n_x)),
        "bc": np.zeros((n_a, 1)),
        "Wo": rand((n_a, n_a + n_x)),
        "bo": np.zeros((n_a, 1)),
        "Wy": rand((n_y, n_a)),
        "by": np.zeros((n_y, 1)),
    }


def lstm_cell_step(xt, a_prev, c_prev, parameters):
    """
    Implements a single forward step of the LSTM-cell with batched input.

    Args:
        xt (ndarray): Input at time t, shape (n_x, m)
        a_prev (ndarray): Hidden state at t-1, shape (n_a, m)
        c_prev (ndarray): Cell state at t-1, shape (n_a, m)
        parameters (dict): LSTM parameters

    Returns:
        a_next (ndarray): Next hidden state, shape (n_a, m)
        c_next (ndarray): Next cell state, shape (n_a, m)
        yt_pred (ndarray): Prediction at time t, shape (n_y, m)
        cache (tuple): For backward pass
    """
    # Retrieve parameters
    Wf, bf = parameters["Wf"], parameters["bf"]
    Wi, bi = parameters["Wi"], parameters["bi"]
    Wc, bc = parameters["Wc"], parameters["bc"]
    Wo, bo = parameters["Wo"], parameters["bo"]
    Wy, by = parameters["Wy"], parameters["by"]

    # Concatenate a_prev and xt along rows
    concat = np.concatenate((a_prev, xt), axis=0)  # (n_a + n_x, m)

    # Compute gates and cell state
    ft = sigmoid(np.dot(Wf, concat) + bf)  # Forget gate
    it = sigmoid(np.dot(Wi, concat) + bi)  # Update gate
    cct = np.tanh(np.dot(Wc, concat) + bc)  # Candidate cell state
    c_next = ft * c_prev + it * cct  # Final cell state
    ot = sigmoid(np.dot(Wo, concat) + bo)  # Output gate
    a_next = ot * np.tanh(c_next)  # Hidden state

    # Compute prediction
    yt_pred = softmax(np.dot(Wy, a_next) + by)  # Output (softmax)

    # Store cache for backward
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    Run an LSTM over Tx time steps.

    Args:
        x (ndarray): Input data, shape (n_x, m, T_x)
        a0 (ndarray): Initial hidden state, shape (n_a, m)
        parameters (dict): LSTM parameters

    Returns:
        a (ndarray): Hidden states, shape (n_a, m, T_x)
        y (ndarray): Predictions, shape (n_y, m, T_x)
        caches (list): List of caches for backprop
    """
    caches = []

    n_x, m, T_x = x.shape
    n_a, _ = a0.shape
    n_y, _ = parameters["Wy"].shape

    # Initialize outputs
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next
    a_next = a0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        xt = x[:, :, t]  # Slice x at time step t → shape (n_x, m)

        a_next, c_next, yt_pred, cache = lstm_cell_step(xt, a_next, c_next, parameters)

        # Store into output tensors
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt_pred

        caches.append(cache)

    return a, y, caches


def lstm_step_backward(da_next, dc_next, cache):
    """
    Implements the backward pass for a single time step of an LSTM.

    Args:
        da_next (ndarray): Gradient of loss w.r.t. next hidden state, shape (n_a, m)
        dc_next (ndarray): Gradient of loss w.r.t. next cell state, shape (n_a, m)
        cache (tuple): Values from the forward pass (used to compute gradients)

    Returns:
        gradients (dict): Dictionary with the following keys:
            - dxt: Gradient of input x_t, shape (n_x, m)
            - da_prev: Gradient of previous hidden state a_{t-1}, shape (n_a, m)
            - dc_prev: Gradient of previous cell state c_{t-1}, shape (n_a, m)
            - dWf, dWi, dWc, dWo: Gradients of weight matrices
            - dbf, dbi, dbc, dbo: Gradients of biases
    """
    # Unpack values from cache
    a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters = cache

    # Retrieve weights
    Wf, Wi, Wc, Wo = (
        parameters["Wf"],
        parameters["Wi"],
        parameters["Wc"],
        parameters["Wo"],
    )

    # Get dimensions
    n_x, m = xt.shape
    n_a, _ = a_next.shape

    # Compute derivatives for gate outputs (Equations 7–10)
    tanh_c_next = np.tanh(c_next)
    dtanh_c_next = 1 - tanh_c_next**2

    # Equation (7): ∂L/∂γ_o
    dot = da_next * tanh_c_next * ot * (1 - ot)

    # Shared intermediate (used for ∂L/∂γ_f, ∂L/∂γ_i, ∂L/∂c̃)
    dc_combined = dc_next + da_next * ot * dtanh_c_next

    # Equation (8): ∂L/∂c̃
    dcct = dc_combined * it * (1 - cct**2)

    # Equation (9): ∂L/∂γ_i
    dit = dc_combined * cct * it * (1 - it)

    # Equation (10): ∂L/∂γ_f
    dft = dc_combined * c_prev * ft * (1 - ft)

    # Concatenate a_prev and x_t
    concat = np.concatenate((a_prev, xt), axis=0)  # shape (n_a + n_x, m)

    # Compute weight gradients (Equations 11–14)
    dWf = np.dot(dft, concat.T)
    dWi = np.dot(dit, concat.T)
    dWc = np.dot(dcct, concat.T)
    dWo = np.dot(dot, concat.T)

    # Compute bias gradients (Equations 15–18)
    dbf = np.sum(dft, axis=1, keepdims=True)
    dbi = np.sum(dit, axis=1, keepdims=True)
    dbc = np.sum(dcct, axis=1, keepdims=True)
    dbo = np.sum(dot, axis=1, keepdims=True)

    # Backprop through [a_prev; x_t] (Equations 19 and 21)
    dconcat = (
        np.dot(Wf.T, dft) + np.dot(Wi.T, dit) + np.dot(Wc.T, dcct) + np.dot(Wo.T, dot)
    )

    da_prev = dconcat[:n_a, :]  # First n_a rows for a_prev (Equation 19)
    dxt = dconcat[n_a:, :]  # Last n_x rows for x_t (Equation 21)

    # Backprop cell state (Equation 20)
    dc_prev = dc_combined * ft

    # Package gradients
    gradients = {
        "dxt": dxt,
        "da_prev": da_prev,
        "dc_prev": dc_prev,
        "dWf": dWf,
        "dbf": dbf,
        "dWi": dWi,
        "dbi": dbi,
        "dWc": dWc,
        "dbc": dbc,
        "dWo": dWo,
        "dbo": dbo,
    }

    return gradients


def lstm_backwards(da, caches):
    """
    Implements the backward pass over an entire sequence for an LSTM.

    Args:
        da (ndarray): Gradients of loss w.r.t. hidden states for all time steps,
                      shape (n_a, m, T_x)
        caches (tuple): Tuple containing:
            - list of caches from each time step (one per step, from lstm_step_forward)
            - input x used in lstm_forward, shape (n_x, m, T_x)

    Returns:
        gradients (dict): Dictionary containing:
            - dx: Gradient of inputs, shape (n_x, m, T_x)
            - da0: Gradient of initial hidden state, shape (n_a, m)
            - dWf, dWi, dWc, dWo: Gradients of weight matrices
            - dbf, dbi, dbc, dbo: Gradients of biases
    """
    # Unpack caches
    lstm_caches, x = caches
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = lstm_caches[0]

    # Retrieve dimensions
    n_a, m, T_x = da.shape
    n_x, _ = xt.shape

    # Initialize gradients
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))

    dWf = np.zeros_like(parameters["Wf"])
    dWi = np.zeros_like(parameters["Wi"])
    dWc = np.zeros_like(parameters["Wc"])
    dWo = np.zeros_like(parameters["Wo"])
    dbf = np.zeros_like(parameters["bf"])
    dbi = np.zeros_like(parameters["bi"])
    dbc = np.zeros_like(parameters["bc"])
    dbo = np.zeros_like(parameters["bo"])

    # Loop backward through time
    for t in reversed(range(T_x)):
        da_curr = da[:, :, t] + da_prevt  # Add current da and carried-over gradient
        dc_curr = dc_prevt

        grads = lstm_step_backward(da_curr, dc_curr, lstm_caches[t])

        # Store per-step gradients
        dx[:, :, t] = grads["dxt"]
        da_prevt = grads["da_prev"]
        dc_prevt = grads["dc_prev"]

        # Accumulate parameter gradients
        dWf += grads["dWf"]
        dWi += grads["dWi"]
        dWc += grads["dWc"]
        dWo += grads["dWo"]
        dbf += grads["dbf"]
        dbi += grads["dbi"]
        dbc += grads["dbc"]
        dbo += grads["dbo"]

    # Final da0 from the last timestep's da_prev
    da0 = da_prevt

    # Pack final gradients
    gradients = {
        "dx": dx,
        "da0": da0,
        "dWf": dWf,
        "dbf": dbf,
        "dWi": dWi,
        "dbi": dbi,
        "dWc": dWc,
        "dbc": dbc,
        "dWo": dWo,
        "dbo": dbo,
    }

    return gradients
