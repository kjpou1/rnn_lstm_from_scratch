"""
Character-Level RNN Training Script (Mini-batch version).

This script trains a simple RNN (built manually with NumPy) using mini-batches instead of single examples.

Features:
- Epoch-based training with batches
- Optional deterministic shuffling for reproducibility
- Sampling after every epoch
- Temperature-controlled text generation

Usage:
    python scratch_char_level_rnn_batch_train.py --dataset dinos --epochs 10 --batch_size 32 --learning_rate 0.01
"""

import argparse

import numpy as np

from data_prep import load_dataset
from optimizers.adam_optimizer import AdamOptimizer
from optimizers.momentum_optimizer import MomentumOptimizer
from optimizers.rmsprop_optimizer import RMSPropOptimizer
from optimizers.sgd_optimizer import SGDOptimizer
from rnn_model import (
    initialize_rnn_parameters,
    rnn_backward,
    rnn_cell_step,
    rnn_forward,
)
from tokenizer import CharTokenizer
from utils import (
    clip,
    cross_entropy_loss,
    get_initial_loss,
    pad_sequences,
    sample_from_logits,
    set_random_seed,
    smooth,
    softmax,
)


def batchify(X, Y, batch_size, seed=None):
    """
    Generator to yield batches from (X, Y).

    Args:
        X (ndarray): Input sequences.
        Y (ndarray): Target sequences.
        batch_size (int): Number of examples per batch.
        seed (int, optional): Seed for deterministic shuffling.

    Yields:
        Tuple of (batch_X, batch_Y).
    """
    m = X.shape[0]
    indices = np.arange(m)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    for start_idx in range(0, m, batch_size):
        end_idx = min(start_idx + batch_size, m)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], Y[batch_idx]


def generate_text(
    parameters, tokenizer, start_string="", temperature=1.0, max_length=50, seed=0
):
    """
    Generate text using a manually implemented character-level RNN model.

    This function samples one character at a time using rnn_cell_step().
    It optionally takes a priming string and temperature scaling for control.

    Args:
        parameters (dict): Trained RNN parameters from scratch.
        tokenizer (CharTokenizer): Tokenizer with character-to-index mapping.
        start_string (str): Text to prime the model with.
        temperature (float): Temperature scaling for randomness.
        max_length (int): Max length of the generated sequence.
        seed (int): Random seed for reproducibility.

    Returns:
        str: Generated character sequence.
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
            a_prev, _, _ = rnn_cell_step(x, a_prev, parameters)

    idx = None
    newline_idx = tokenizer.char_to_ix["\n"]
    counter = 0
    np.random.seed(seed)

    while idx != newline_idx and counter < max_length:
        a_prev, y_pred, _ = rnn_cell_step(x, a_prev, parameters)

        logits = np.log(y_pred + 1e-9) / temperature
        probs = softmax(logits)

        idx = sample_from_logits(logits)
        generated_indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        counter += 1

    generated_text = tokenizer.sequences_to_texts(generated_indices)
    return start_string + generated_text


def get_optimizer(name, learning_rate):
    """
    Helper to instantiate the selected optimizer.
    """
    if name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate)
    elif name == "momentum":
        return MomentumOptimizer(learning_rate=learning_rate, beta=0.5)
    elif name == "rms":
        return RMSPropOptimizer(learning_rate=learning_rate, beta=0.9)
    elif name == "adam":
        return AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
    else:
        raise ValueError(f"Unsupported optimizer: {name}. Choose from ['sgd']")


def train_model(
    X,
    Y,
    vocab_size,
    tokenizer,
    n_a=50,
    epochs=10,
    batch_size=32,
    learning_rate=0.01,
    optimizer_name="sgd",
    temperature=1.0,
    seq_length=50,
    clip_value=5.0,
    deterministic=False,
):
    """
    Train the scratch RNN model with batch updates.
    """
    parameters = initialize_rnn_parameters(n_a, vocab_size, vocab_size)
    best_parameters = None  # <-- NEW
    loss = get_initial_loss(vocab_size, len(X))
    best_loss = float("inf")

    optimizer = get_optimizer(optimizer_name, learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in batchify(
            X, Y, batch_size, seed=epoch if deterministic else None
        ):
            for x_seq, y_seq in zip(batch_x, batch_y):
                cache = rnn_forward(x_seq, np.zeros((n_a, 1)), parameters)
                y_hat, *_ = cache
                gradients, a = rnn_backward(x_seq, y_seq, parameters, cache)
                gradients = clip(gradients, maxValue=clip_value)
                parameters = optimizer.update(parameters, gradients)

                curr_loss = cross_entropy_loss(y_hat, y_seq)
                loss = smooth(loss, curr_loss)

                epoch_loss += curr_loss
                num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Save best parameters
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_parameters = {k: np.copy(v) for k, v in parameters.items()}  # <-- COPY

        print(f"\nEpoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        print(f"--- Sampling after Epoch {epoch+1} ---")
        for idx in range(3):
            generated_text = generate_text(
                parameters,
                tokenizer,
                start_string="",
                temperature=temperature,
                max_length=seq_length,
                seed=epoch * 3 + idx,
            )
            sample_name = generated_text[0].upper() + generated_text[1:]
            print(sample_name)

    return best_parameters  # <-- RETURN the best one


def main():
    parser = argparse.ArgumentParser(description="Train scratch RNN with mini-batches")
    parser.add_argument("--dataset", type=str, default="dinos", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "rms", "adam"],
        help="Optimizer type (default: 'sgd')",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--clip_value", type=float, default=5.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Deterministic shuffling for reproducibility",
    )

    args = parser.parse_args()

    print("Training Parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Load and prepare data
    corpus, tokenizer, X, Y = load_dataset(
        args.dataset, mode="line_by_line", lowercase=True
    )

    # Pad sequences
    X = pad_sequences(X, padding="post")
    Y = pad_sequences(Y, padding="post")

    X = np.array(X)
    Y = np.array(Y)

    vocab_size = tokenizer.vocab_size()

    # Train
    parameters = train_model(
        X,
        Y,
        vocab_size,
        tokenizer,
        n_a=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        temperature=args.temperature,
        seq_length=args.seq_length,
        clip_value=args.clip_value,
        deterministic=args.deterministic,
    )

    # Print the final loss
    print("Training complete")
    # The number of dinosaur names to print
    seed = 0
    dino_names = 7
    print("\n--- Generating samples:")
    for name in range(dino_names):
        generated_text = generate_text(
            parameters,
            tokenizer,
            start_string="",
            temperature=args.temperature,
            max_length=args.seq_length,
            seed=seed,
        )
        last_dino_name = generated_text[0].upper() + generated_text[1:]
        print(last_dino_name)

        seed += 1


if __name__ == "__main__":
    main()
