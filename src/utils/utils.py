import copy
import random

import numpy as np
import tensorflow as tf


def softmax(z):
    """
    Compute the softmax activation in a numerically stable way.

    This function supports both single examples and batched inputs.

    Args:
        z (np.ndarray): Logits of shape (n_y, m) or (n_y, 1),
                        where n_y is number of classes and m is batch size.

    Returns:
        np.ndarray: Softmax output with same shape as input (n_y, m)
    """
    axis = 0 if z.ndim == 2 else (0 if z.shape[1] == 1 else 1)
    # 🧠 Numerical stability trick:
    # Subtract the max value from each column to avoid large exponentials (overflow)
    # This preserves relative differences while stabilizing the computation
    z_max = np.max(z, axis=axis, keepdims=True)  # shape: (1, m)

    # Compute exponentials after centering
    exp_z = np.exp(z - z_max)

    # Normalize by the sum across classes (axis=0 handles vertical softmax over classes)
    softmax_output = exp_z / np.sum(exp_z, axis=axis, keepdims=True)

    return softmax_output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def cross_entropy_loss(z, y_true):
    """
    Stable cross-entropy loss from logits.

    Args:
        z (ndarray): Raw logits of shape (vocab_size, T_x) or (vocab_size, 1, T_x)
        y_true (list[int]): Ground truth token indices

    Returns:
        float: Average cross-entropy loss
    """
    # If 3D: squeeze out middle dimension (assumes shape (vocab_size, 1, T_x))
    if z.ndim == 3 and z.shape[1] == 1:
        z = np.squeeze(z, axis=1)  # → (vocab_size, T_x)

    T_x = len(y_true)
    logits = z - np.max(z, axis=0, keepdims=True)  # for numerical stability
    exp_logits = np.exp(logits)
    log_probs = logits - np.log(
        np.sum(exp_logits, axis=0, keepdims=True)
    )  # log softmax

    loss = 0.0
    for t, target_index in enumerate(y_true):
        loss -= log_probs[target_index, t]

    return loss / T_x


def cross_entropy_loss_grad_from_logits(logits, y_true):
    """
    Compute gradient of the cross-entropy loss with respect to logits.

    Args:
        logits (ndarray): Raw logits, shape (vocab_size, 1, T_x)
        y_true (list[int]): Ground truth token indices, length T_x

    Returns:
        dy (ndarray): Gradient w.r.t logits, same shape as logits
    """
    y_hat = softmax(logits)  # shape (vocab_size, 1, T_x)
    dy = y_hat.copy()

    for t, target_index in enumerate(y_true):
        dy[target_index, 0, t] -= 1  # softmax - one-hot at each time step

    dy /= len(y_true)  # average over sequence length
    return dy


def clip(gradients, maxValue):
    """
    Clip all gradients in the dictionary to [-max_value, max_value].

    Args:
        gradients (dict): Dictionary of gradients (e.g., from backprop)
        max_value (float): Threshold for clipping

    Returns:
        dict: Same dictionary with clipped gradients
    """
    clipped = {}
    for key, grad in gradients.items():
        clipped[key] = np.clip(grad, -maxValue, maxValue)
    return clipped


def sample_from_logits(logits, seed=None):
    """
    Sample an index from raw logits (no softmax yet).

    Args:
        logits (ndarray): Array of logits, shape (vocab_size, 1)
        seed (int): Optional random seed for reproducibility

    Returns:
        idx (int): Sampled index
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Apply softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))  # stability trick
    probs = exp_logits / np.sum(exp_logits)

    # Step 2: Sample from the probability distribution
    idx = np.random.choice(range(len(probs)), p=probs.ravel())

    return idx


def get_sample(sample_ix, ix_to_char):
    txt = "".join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    return txt


def pad_sequences(sequences, padding="post", value=0):
    """
    Pads a list of variable-length sequences to the same length.

    Args:
        sequences (list of lists): List of sequences (list of ints).
        padding (str): "pre" or "post" (where to add padding). Default is "post".
        value (int): Padding value to use (default 0).

    Returns:
        np.ndarray: 2D array with padded sequences.
    """

    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    padded = np.full((batch_size, max_len), value, dtype=np.int32)

    for i, seq in enumerate(sequences):
        if padding == "post":
            padded[i, : len(seq)] = seq
        elif padding == "pre":
            padded[i, -len(seq) :] = seq
        else:
            raise ValueError(f"Unsupported padding strategy: {padding}")

    return padded


def set_random_seed(seed):
    """
    Sets random seed for Python random, NumPy, and TensorFlow for full reproducibility.
    Args:
        seed (int): Random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
