"""
Character-Level RNN Training Script (from scratch).

This script trains a simple RNN (built manually with NumPy) on a character-level dataset, such as dinosaur names,
using standard RNN forward and backward propagation.

Features:
- Preprocessing:
  - Loads and tokenizes character-level datasets (e.g., "dinos.txt").
  - Creates input-target sequences for supervised learning.

- Training Loop:
  - Performs forward and backward passes through the RNN.
  - Applies gradient clipping to prevent exploding gradients.
  - Updates parameters using vanilla SGD.
  - Smooths the loss for better tracking over time.

- Sampling:
  - Generates new character sequences by sampling from the model.
  - Supports temperature scaling to control creativity during generation.

- Command-Line Interface:
  - Allows dynamic adjustment of dataset, number of iterations, learning rate, temperature, hidden size, etc.

Usage:
    python scratch_char_level_rnn_model.py --dataset dinos --iterations 10000 --temperature 1.0 --hidden_size 50

Arguments:
- --dataset: Name of the dataset file (default: "dinos.txt")
- --iterations: Number of training iterations
- --learning_rate: Learning rate for gradient descent
- --temperature: Sampling temperature (>1 more random, <1 more deterministic)
- --hidden_size: Number of hidden units in the RNN
- --sample_every: Interval (iterations) to sample text during training
- --seq_length: Maximum length of generated sequences
- --clip_value: Maximum allowed gradient norm for clipping

Notes:
- No external deep learning libraries (TensorFlow, PyTorch) are used.
- Relies solely on NumPy for matrix operations.
- Implements basic utilities such as softmax, loss smoothing, and sampling from logits.

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
    smooth,
    softmax,
)


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


def main(
    dataset_name="dinos",
    n_a=50,
    num_iterations=10000,
    learning_rate=0.01,
    optimizer_name="sgd",
    temperature=1.0,
    sample_every=1000,
    seq_length=25,
    clip_value=5.0,
    verbose=False,
):
    dino_names = 7  # Number of dinosaur names to print

    _, tokenizer, X, Y = load_dataset("dinos", mode="line_by_line", lowercase=True)

    X = pad_sequences(X, padding="post")
    Y = pad_sequences(Y, padding="post")

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    vocab_size = tokenizer.vocab_size()
    # Initialize RNN parameters
    parameters = initialize_rnn_parameters(n_a, vocab_size, vocab_size)
    a_prev = np.zeros((n_a, 1))

    # Training loop
    loss = get_initial_loss(vocab_size, len(X))
    best_loss = float("inf")

    optimizer = get_optimizer(optimizer_name, learning_rate)

    last_dino_name = "abc"

    for iteration in range(num_iterations):
        # === 1. Fetch training sample ===
        idx = iteration % len(X)
        x_seq = X[idx]  # input sequence (list of token indices)
        y_seq = Y[idx]  # target sequence (same length, next char indices)

        # === Clean Path: Forward → Loss + Grad → Backward → Update ===

        # === 2. Forward pass ===
        cache = rnn_forward(x_seq, a_prev, parameters)
        _, _, logits, _ = cache  # logits: shape (vocab_size, T_x)

        # === 3. Compute loss and ∂L/∂logits ===
        curr_loss, dy = compute_loss_and_grad(logits, y_seq, reduction="mean")

        # === 4. Project ∂L/∂logits → ∂L/∂a for RNN backprop ===
        da = project_logit_grad_to_hidden(dy, parameters["Wya"])

        # === 5. Backward pass through RNN ===
        gradients, a = rnn_backward(da, parameters, cache)

        # === 6. Output layer gradients (∂L/∂Wya, ∂L/∂by) ===
        grads_out = compute_output_layer_gradients(dy, a)
        gradients.update(grads_out)

        # === 7. Clip gradients to mitigate exploding gradients ===
        gradients = clip(gradients, maxValue=clip_value)

        # === 8. Parameter update step ===
        parameters = optimizer.update(parameters, gradients)

        # === 9. Track smoothed loss (EMA) and best loss ===
        loss = smooth(loss, curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss

        # === 10. Track gradient norm (for logging/debugging) ===
        gradient_norms = sum(
            np.linalg.norm(gradients[g]) for g in gradients if g.startswith("d")
        )

        # === 11. Carry final hidden state forward ===
        a_prev = a[len(x_seq) - 1]

        # === 12. Periodic sampling ===
        if iteration % sample_every == 0:
            print(
                f"\nIteration {iteration} - Raw Loss: {curr_loss:.4f} | Smoothed Loss: {loss:.4f} | Best Loss: {best_loss:.4f} | Grad Norm: {gradient_norms:.2f}"
            )
            print(f"\n--- Sampling after iteration: {iteration} ---")
            for i in range(3):
                generated_text = generate_text(
                    parameters,
                    tokenizer,
                    start_string="",
                    temperature=temperature,
                    max_length=seq_length,
                    seed=iteration + i,
                )
                print(generated_text.capitalize())

    # Print the final loss
    print("Training complete")
    # The number of dinosaur names to print
    seed = 0
    print("\n--- Generating samples:")
    for name in range(dino_names):
        generated_text = generate_text(
            parameters,
            tokenizer,
            start_string="",
            temperature=temperature,
            max_length=seq_length,
            seed=seed,
        )
        last_dino_name = generated_text[0].upper() + generated_text[1:]
        print(last_dino_name)

        seed += 1


# +-------+-------------------------------------------+
# | Temp  | Effect                                    |
# +-------+-------------------------------------------+
# | < 1   | Sharper, more deterministic, less creative |
# | = 1   | Normal softmax behavior (standard sampling) |
# | > 1   | Flatter, more exploratory, more creative   |
# +-------+-------------------------------------------+

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN or LSTM model")
    parser.add_argument("--dataset", type=str, default="dinos", help="Dataset name")
    parser.add_argument("--iterations", type=int, default=10000, help="Training steps")
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
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--seq_length", type=int, default=25)
    parser.add_argument("--clip_value", type=float, default=5.0)

    args = parser.parse_args()

    print("Training Parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    main(
        dataset_name=args.dataset,
        n_a=args.hidden_size,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        temperature=args.temperature,
        sample_every=args.sample_every,
        seq_length=args.seq_length,
        clip_value=args.clip_value,
        verbose=True,
    )
