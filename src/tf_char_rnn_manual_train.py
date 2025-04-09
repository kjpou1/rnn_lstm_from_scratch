import argparse
import random
import tensorflow as tf
from utils import pad_sequences, set_random_seed
from tf_char_rnn import TFCharRNN
from tokenizer import CharTokenizer
import numpy as np
from data_prep import load_dataset

def generate_text(model, start_string, tokenizer, max_length=50, temperature=1.0, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)

    if not start_string:
        start_string = random.choice(tokenizer.get_vocab()[1:])

    input_indices = tokenizer.texts_to_sequences(start_string)
    input_indices = tf.expand_dims(input_indices, 0)

    generated_indices = []
    states = None

    for _ in range(max_length):
        predictions, states = model(input_indices, states=states, return_state=True)
        last_logits = predictions[:, -1, :]
        scaled_logits = last_logits / temperature
        probs = tf.nn.softmax(scaled_logits)

        predicted_id = tf.random.categorical(tf.math.log(probs), num_samples=1)
        predicted_id = tf.squeeze(predicted_id, axis=-1)

        generated_indices.append(predicted_id.numpy()[0])
        input_indices = tf.expand_dims(predicted_id, 0)

        if tokenizer.ix_to_char[predicted_id.numpy()[0]] == "\n":
            break

    generated_text = tokenizer.sequences_to_texts(generated_indices)
    return start_string + generated_text

def batchify(X, Y, batch_size, shuffle=True):
    """Generator that yields batches, optionally shuffling data."""
    m = X.shape[0]
    indices = np.arange(m)
    if shuffle:
        np.random.shuffle(indices)
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)

    for start_idx in range(0, m, batch_size):
        end_idx = min(start_idx + batch_size, m)
        batch_idx = indices[start_idx:end_idx]
        yield tf.gather(X, batch_idx), tf.gather(Y, batch_idx)


def loss_fn(labels, logits):
    """Sparse categorical crossentropy."""
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )


def train_model(X, Y, vocab_size, tokenizer, epochs=10, batch_size=32, lr=0.001):
    model = TFCharRNN(vocab_size, embedding_dim=64, rnn_units=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in batchify(X, Y, batch_size):
            with tf.GradientTape() as tape:
                logits = model(batch_x, training=True)
                loss = loss_fn(batch_y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += loss.numpy()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # ✨ Sample after every epoch
        print(f"\n--- Sampling after epoch {epoch+1} ---")
        for _ in range(3):  # Generate 3 samples
            generated_text = generate_text(
                model,
                start_string="",   # or pick a letter like "a"
                tokenizer=tokenizer,
                temperature=1.0,
                max_length=50,
            )
            sample_name = generated_text[0].upper() + generated_text[1:]
            print(sample_name)
        print("\n")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic runs")
    args = parser.parse_args()

    if args.deterministic and args.seed is not None:
        print(f"Setting global random seed to {args.seed}")
        set_random_seed(args.seed)
    else:
        print("Running with full randomness")

    # Load data, train, etc.
    corpus, tokenizer, X, Y = load_dataset("dinos", mode="line_by_line", lowercase=True)

    # Pad
    X = pad_sequences(X, padding="post")
    Y = pad_sequences(Y, padding="post")

    X = tf.constant(X, dtype=tf.int32)
    Y = tf.constant(Y, dtype=tf.int32)

    # Model training
    model = train_model(X, Y, tokenizer.vocab_size(), tokenizer, epochs=10)

    # Sampling
    for _ in range(5):
        text = generate_text(model, start_string="", tokenizer=tokenizer, seed=args.seed if args.deterministic else None)
        text = text[0].upper() + text[1:]
        print(text)
