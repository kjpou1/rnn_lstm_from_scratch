"""
Character-Level LSTM Training Script (from scratch).

This script trains a simple LSTM (built manually with NumPy) on a character-level dataset.

Usage:
    python scratch_char_level_lstm_model.py --dataset dinos --iterations 10000 --temperature 1.0 --hidden_size 50
"""

import argparse

import numpy as np

from src.utils.grad_utils import compute_output_layer_gradients
from src.utils.loss_utils import compute_loss_and_grad, project_logit_grad_to_hidden

from .data_prep import load_dataset
from .models.lstm_model import initialize_lstm_parameters, lstm_backwards, lstm_forward
from .optimizers.adam_optimizer import AdamOptimizer
from .optimizers.momentum_optimizer import MomentumOptimizer
from .optimizers.rmsprop_optimizer import RMSPropOptimizer
from .optimizers.sgd_optimizer import SGDOptimizer
from .sampling import generate_text_lstm
from .tokenizer import CharTokenizer
from .utils.utils import (
    clip,
    cross_entropy_loss,
    cross_entropy_loss_grad_from_logits,
    get_initial_loss,
    pad_sequences,
    smooth,
    softmax,
)


def get_optimizer(name, learning_rate):
    if name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate)
    elif name == "momentum":
        return MomentumOptimizer(learning_rate=learning_rate, beta=0.5)
    elif name == "rms":
        return RMSPropOptimizer(learning_rate=learning_rate, beta=0.9)
    elif name == "adam":
        return AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


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
    dino_names = 7

    _, tokenizer, X, Y = load_dataset(dataset_name, mode="line_by_line", lowercase=True)

    X = pad_sequences(X, padding="post")
    Y = pad_sequences(Y, padding="post")

    vocab_size = tokenizer.vocab_size()
    parameters = initialize_lstm_parameters(n_a, vocab_size, vocab_size)

    a_prev = np.zeros((n_a, 1))

    loss = get_initial_loss(vocab_size, len(X))
    best_loss = float("inf")

    optimizer = get_optimizer(optimizer_name, learning_rate)

    for iteration in range(num_iterations):
        # Select the next training example (cyclically through the dataset)
        idx = iteration % len(X)
        x_seq = X[idx]  # Input sequence (list of indices)
        y_seq = Y[idx]  # Target sequence (list of indices)

        # === Clean Path: Forward → Loss + Grad → Backward → Update ===

        # 1. Forward pass through the LSTM
        a, logits, caches = lstm_forward(x_seq, a_prev, parameters)

        # 2. Compute cross-entropy loss and gradient w.r.t. logits (∂L/∂z)
        curr_loss, dy = compute_loss_and_grad(logits, y_seq)

        # 3. Project output gradients back to hidden states (∂L/∂a)
        da = project_logit_grad_to_hidden(dy, parameters["Wy"])

        # 4. Backward pass through the LSTM (∂L/∂params from da)
        gradients = lstm_backwards(da, caches)

        # 5. Compute gradients w.r.t. output layer parameters (dWy, dby)
        grads_out = compute_output_layer_gradients(dy, a)
        gradients.update(grads_out)

        # 6. Clip gradients to mitigate exploding gradients
        gradients = clip(gradients, maxValue=clip_value)

        # 7. Update parameters using chosen optimizer
        parameters = optimizer.update(parameters, gradients)

        # 8. Update running smoothed loss (exponential moving average)
        loss = smooth(loss, curr_loss)

        # 9. Update best loss tracker
        if curr_loss < best_loss:
            best_loss = curr_loss

        # 10. Compute gradient norm (for diagnostics/debugging)
        grad_norm = sum(
            np.linalg.norm(gradients[k]) for k in gradients if k.startswith("d")
        )

        # 11. Carry forward final hidden state for next iteration
        a_prev = a[:, :, -1]

        # 12. Logging and sampling
        if iteration % sample_every == 0:
            print(
                f"\nIteration {iteration} - Raw Loss: {curr_loss:.4f} | Smoothed Loss: {loss:.4f} | Best: {best_loss:.4f} | Grad Norm: {grad_norm:.2f}"
            )
            print(f"--- Sampling after iteration {iteration} ---")
            for s in range(3):
                gen_text = generate_text_lstm(
                    parameters,
                    tokenizer,
                    start_string="",
                    temperature=temperature,
                    max_length=seq_length,
                    seed=iteration + s,
                )
                print(gen_text.capitalize())

    print("\n--- Final Samples ---")
    for i in range(dino_names):
        gen_text = generate_text_lstm(
            parameters=parameters,
            tokenizer=tokenizer,
            start_string="",
            temperature=temperature,
            max_length=seq_length,
            seed=i,
        )
        print(gen_text.capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a character-level LSTM (NumPy)")
    parser.add_argument("--dataset", type=str, default="dinos")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "rms", "adam"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--seq_length", type=int, default=25)
    parser.add_argument("--clip_value", type=float, default=5.0)
    args = parser.parse_args()

    print("Training Parameters:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

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
    )
