import numpy as np

from src.activations.sigmoid import SigmoidActivation
from src.activations.softmax import SoftmaxActivation
from src.activations.tanh import TanhActivation
from src.utils.utils import sigmoid, softmax


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
    zf = np.dot(Wf, concat) + bf  # Forget gate
    ft = SigmoidActivation.forward(zf)  # Forget gate
    zi = np.dot(Wi, concat) + bi  # Input gate
    it = SigmoidActivation.forward(zi)  # Input gate
    cctf = np.dot(Wc, concat) + bc  # Candidate cell state
    cct = TanhActivation.forward(cctf)  # Candidate cell state
    c_next = ft * c_prev + it * cct  # Final cell state
    zo = np.dot(Wo, concat) + bo  # Output gate
    ot = SigmoidActivation.forward(zo)  # Output gate
    a_next = ot * TanhActivation.forward(c_next)  # Hidden state

    # Compute prediction
    logits = np.dot(Wy, a_next) + by  # Linear output

    # Store cache for backward
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, logits, cache


def lstm_forward(x_seq, a0, parameters):
    """
    Run an LSTM over T_x time steps.

    Args:
        x_seq (ndarray): Sequence of token indices, shape (T_x,)
        a0 (ndarray): Initial hidden state, shape (n_a, 1)
        parameters (dict): LSTM parameters

    Returns:
        a (ndarray): Hidden states, shape (n_a, 1, T_x)
        logits (ndarray): Raw output scores, shape (n_y, 1, T_x)
        caches (list): List of caches for backprop
    """
    caches = []

    n_a, _ = a0.shape
    n_y, _ = parameters["Wy"].shape
    T_x = len(x_seq)

    # Infer vocab size from parameter shapes
    n_a_check, n_concat = parameters["Wf"].shape
    assert n_a_check == n_a, "Mismatch between a0 and Wf shape"
    n_x = n_concat - n_a
    vocab_size = n_x

    a = np.zeros((n_a, 1, T_x))
    c = np.zeros((n_a, 1, T_x))
    logits = np.zeros((n_y, 1, T_x))

    a_next = a0
    c_next = np.zeros_like(a0)

    for t in range(T_x):
        x_t = np.zeros((vocab_size, 1))
        x_t[x_seq[t]] = 1  # one-hot

        a_next, c_next, logits_t, cache = lstm_cell_step(
            x_t, a_next, c_next, parameters
        )

        a[:, :, t] = a_next
        c[:, :, t] = c_next
        logits[:, :, t] = logits_t
        caches.append(cache)

    return a, logits, caches


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
    tanh_c_next = TanhActivation.forward(c_next)
    dtanh_c_next = TanhActivation.backward(c_next)

    # Equation (7): ∂L/∂γ_o
    dot = da_next * tanh_c_next * SigmoidActivation.backward(ot)

    # Shared intermediate (used for ∂L/∂γ_f, ∂L/∂γ_i, ∂L/∂c̃)
    dc_combined = dc_next + da_next * ot * dtanh_c_next

    # Equation (8): ∂L/∂c̃
    dcct = dc_combined * it * TanhActivation.backward(cct)

    # Equation (9): ∂L/∂γ_i
    dit = dc_combined * cct * SigmoidActivation.backward(it)

    # Equation (10): ∂L/∂γ_f
    dft = dc_combined * c_prev * SigmoidActivation.backward(ft)

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
            - list of caches from each time step (from lstm_cell_step)
            - input sequence x used in lstm_forward, shape (n_x, m, T_x)

    Returns:
        gradients (dict): Dictionary containing:
            - dx: Gradient of inputs, shape (n_x, m, T_x)
            - da0: Gradient of initial hidden state, shape (n_a, m)
            - dWf, dWi, dWc, dWo: Gradients of weight matrices
            - dbf, dbi, dbc, dbo: Gradients of biases
    """
    # Unpack caches
    lstm_caches = caches
    (*_, xt, parameters) = lstm_caches[0]

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
        # Current timestep's total gradient of loss w.r.t. hidden state
        da_curr = da[:, :, t] + da_prevt
        dc_curr = dc_prevt

        # Single timestep backward pass
        grads = lstm_step_backward(da_curr, dc_curr, lstm_caches[t])

        # Store input gradient for timestep t
        dx[:, :, t] = grads["dxt"]

        # Carry over gradients for next iteration
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

    # Final da0 from the earliest timestep
    da0 = da_prevt

    # Pack and return all gradients
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
