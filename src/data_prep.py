import os

import numpy as np

from .tokenizer import CharTokenizer


def load_text(path):
    """Reads a text file and returns the full corpus string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def one_hot_encode(sequence, vocab_size):
    """One-hot encodes a sequence of indices."""
    seq_len = len(sequence)
    one_hot = np.zeros((vocab_size, seq_len))
    for t, idx in enumerate(sequence):
        one_hot[idx, t] = 1
    return one_hot


def create_training_sequences_continuous(corpus, tokenizer, seq_length=25):
    """
    Splits corpus into (input, target) pairs for continuous text (e.g., Shakespeare).
    """
    encoded = tokenizer.texts_to_sequences(corpus)
    X, Y = [], []
    for i in range(0, len(encoded) - seq_length):
        X.append(encoded[i : i + seq_length])
        Y.append(encoded[i + 1 : i + seq_length + 1])
    return np.array(X), np.array(Y)


def create_training_sequences_line_by_line(lines, tokenizer):
    """
    Splits corpus into (input, target) pairs for line-by-line text (e.g., Dinos).
    """
    newline_ix = tokenizer.texts_to_sequences("\n")[0]
    X, Y = [], []
    sequences = [tokenizer.texts_to_sequences(name) for name in lines]

    # for seq in sequences:
    #     if len(seq) < 2:
    #         continue
    #     X.append(seq[:-1])  # Input: all chars except last
    #     Y.append(seq[1:])  # Target: shifted by 1 char

    for seq in sequences:
        if len(seq) < 1:
            continue
        X.append(seq)
        Y.append(seq[1:] + [newline_ix])  # shift and add newline token
    return np.array(X, dtype=object), np.array(Y, dtype=object)


def load_dataset(name, mode="continuous", seq_length=25, lowercase=True):
    """
    Loads a specific dataset and returns corpus, tokenizer, and (X, Y) pairs.

    Args:
        name (str): Dataset name. Options: "dinos", "shakespeare"
        mode (str): "continuous" or "line_by_line"
        seq_length (int): Length of input sequence (for continuous mode)
        lowercase (bool): Whether to lowercase the corpus.

    Returns:
        tuple: (corpus, tokenizer, X, Y)
    """
    dataset_paths = {
        "dinos": "./data/dinos.txt",
        "shakespeare": "./data/shakespeare.txt",
    }
    if name not in dataset_paths:
        raise ValueError(
            f"Unknown dataset: {name}. Choose from {list(dataset_paths.keys())}"
        )

    path = dataset_paths[name]
    corpus = load_text(path)
    if lowercase:
        corpus = corpus.lower()

    tokenizer = CharTokenizer(oov_token=None)
    if mode == "continuous":
        tokenizer.fit(corpus)
        X, Y = create_training_sequences_continuous(corpus, tokenizer, seq_length)
    elif mode == "line_by_line":
        lines = corpus.strip().split("\n")
        data = "\n".join(lines)
        tokenizer.fit(data)  # fit on joined data
        X, Y = create_training_sequences_line_by_line(lines, tokenizer)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'continuous' or 'line_by_line'.")

    return corpus, tokenizer, X, Y


if __name__ == "__main__":
    dataset_name = "dinos"
    # or dataset_name = "shakespeare"
    mode = "line_by_line" if dataset_name == "dinos" else "continuous"

    corpus, tokenizer, X, Y = load_dataset(dataset_name, mode=mode)

    print(f"Loaded {dataset_name} corpus with {len(corpus)} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print(f"Generated {len(X)} training sequences")

    print("Example input sequence:", X[0])
    print("Example target sequence:", Y[0])
