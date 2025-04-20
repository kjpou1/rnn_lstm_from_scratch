# sampling.py


import numpy as np

from src.rnn_model import rnn_cell_step
from src.utils import sample_from_logits, softmax


def generate_text(
    parameters, tokenizer, start_string="", temperature=1.0, max_length=50, seed=0
):
    """
    Generate text using a manually implemented character-level RNN model.
    """
    vocab_size = parameters["by"].shape[0]
    n_a = parameters["Waa"].shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    generated_indices = []

    if start_string:
        input_indices = tokenizer.texts_to_sequences(start_string)
        for idx in input_indices:
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            a_prev, *_ = rnn_cell_step(x, a_prev, parameters)

    idx = None
    newline_idx = tokenizer.char_to_ix["\n"]
    counter = 0
    np.random.seed(seed)

    while idx != newline_idx and counter < max_length:
        a_prev, logits, *_ = rnn_cell_step(x, a_prev, parameters)

        scaled_logits = logits / temperature
        probs = softmax(scaled_logits)
        idx = sample_from_logits(scaled_logits)

        generated_indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        counter += 1

    generated_text = tokenizer.sequences_to_texts(generated_indices)
    return start_string + generated_text
