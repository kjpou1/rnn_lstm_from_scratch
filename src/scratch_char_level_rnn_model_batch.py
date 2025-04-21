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

from src.sampling import generate_text
from src.utils.grad_utils import compute_output_layer_gradients
from src.utils.loss_utils import compute_loss_and_grad, project_logit_grad_to_hidden

from .data_prep import load_dataset
from .models.rnn_model import (
    initialize_rnn_parameters,
    rnn_backward,
    rnn_cell_step,
    rnn_forward,
)
from .optimizers.adam_optimizer import AdamOptimizer
from .optimizers.momentum_optimizer import MomentumOptimizer
from .optimizers.rmsprop_optimizer import RMSPropOptimizer
from .optimizers.sgd_optimizer import SGDOptimizer
from .tokenizer import CharTokenizer
from .utils.utils import (
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
        # === 1. Initialize epoch tracking ===
        epoch_loss = 0
        num_batches = 0

        # === 2. Iterate over mini-batches (optionally shuffled) ===
        for batch_x, batch_y in batchify(
            X, Y, batch_size, seed=epoch if deterministic else None
        ):
            for x_seq, y_seq in zip(batch_x, batch_y):
                # === 3. Stateless forward pass ===
                a0 = np.zeros((n_a, 1))  # No hidden carryover in batch training
                cache = rnn_forward(x_seq, a0, parameters)
                _, _, logits, _ = cache  # logits shape: (vocab_size, T_x)

                # === 4. Compute loss and ∂L/∂logits ===
                curr_loss, dy = compute_loss_and_grad(logits, y_seq, reduction="sum")

                # === 5. Project ∂L/∂logits to ∂L/∂a for RNN backward
                da = project_logit_grad_to_hidden(dy, parameters["Wya"])

                # === 6. Backward pass through time
                gradients, a = rnn_backward(x_seq, y_seq, parameters, cache)

                # === 7. Compute ∂L/∂Wya and ∂L/∂by from logits
                grads_out = compute_output_layer_gradients(dy, a)
                gradients.update(grads_out)

                # === 8. Clip gradients to prevent exploding updates
                gradients = clip(gradients, maxValue=clip_value)

                # === 9. Apply optimizer step
                parameters = optimizer.update(parameters, gradients)

                # === 10. Track batch loss
                epoch_loss += curr_loss
                num_batches += 1

        # === 11. Average loss for the epoch ===
        avg_loss = epoch_loss / num_batches

        # === 12. Save best model so far ===
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_parameters = {k: np.copy(v) for k, v in parameters.items()}

        # === 13. Logging + sampling output ===
        print(f"\nEpoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

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
            print(generated_text.capitalize())

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
