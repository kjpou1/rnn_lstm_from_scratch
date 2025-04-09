import copy
import random

import numpy as np
import tensorflow as tf


def softmax(x):
    """
    Compute softmax of a vector.

    Args:
        x (ndarray): Input vector

    Returns:
        ndarray: Softmax-normalized probabilities
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def cross_entropy_loss(y_hat, y_true):
    """
    Compute total cross-entropy loss over a sequence.

    Args:
        y_hat (dict): {t: ndarray of shape (vocab_size, 1)} — softmax outputs at each timestep
        y_true (list[int]): List of target indices (length T_x)

    Returns:
        float: total loss
    """
    loss = 0.0
    for t, target_index in enumerate(y_true):
        prob = y_hat[t][target_index, 0]
        # loss -= np.log(prob + 1e-12)  # prevent log(0)
        loss -= np.log(prob)  # prevent log(0)

    return loss


def clip(gradients, maxValue):
    """
    Clips the gradients' values between minimum and maximum.

    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue

    Returns:
    gradients -- a dictionary with the clipped gradients.
    """
    gradients = copy.deepcopy(gradients)

    dWaa, dWax, dWya, dba, dby = (
        gradients["dWaa"],
        gradients["dWax"],
        gradients["dWya"],
        gradients["dba"],
        gradients["dby"],
    )

    # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for grad in [dWaa, dWax, dWya, dba, dby]:
        np.clip(grad, -maxValue, maxValue, out=grad)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}

    return gradients


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
            padded[i, :len(seq)] = seq
        elif padding == "pre":
            padded[i, -len(seq):] = seq
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