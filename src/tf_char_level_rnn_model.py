import random

import tensorflow as tf

from data_prep import load_dataset
from src.utils.utils import pad_sequences
from tf_char_rnn import TFCharRNN
from tokenizer import CharTokenizer


def loss_fn(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


def generate_text(model, start_string, tokenizer, max_length=50, temperature=1.0):
    """
    Generate text using a trained character-level RNN model, avoiding <UNK> sampling.
    """

    # Handle empty start_string
    if not start_string:
        start_string = random.choice(tokenizer.get_vocab()[1:])  # skip <UNK> if present

    # Convert start string to indices
    input_indices = tokenizer.texts_to_sequences(start_string)
    input_indices = tf.expand_dims(input_indices, 0)  # Shape (1, len(start_string))

    generated_indices = []  # To collect predicted indices

    # Reset model states
    states = None

    for _ in range(max_length):
        predictions, states = model(input_indices, states=states, return_state=True)

        # Focus on the last time step's output
        last_logits = predictions[:, -1, :]  # (batch_size, vocab_size)

        # Apply temperature scaling
        scaled_logits = last_logits / temperature
        probs = tf.nn.softmax(scaled_logits)

        # Sample from the probability distribution
        predicted_id = tf.random.categorical(tf.math.log(probs), num_samples=1)
        predicted_id = tf.squeeze(predicted_id, axis=-1)

        # Append the predicted index
        generated_indices.append(predicted_id.numpy()[0])

        # Update input to the model (feed back the prediction)
        input_indices = tf.expand_dims(predicted_id, 0)

        # Optional: stop early if model predicts '\n'
        if tokenizer.ix_to_char[predicted_id.numpy()[0]] == "\n":
            break

    # Convert indices back to characters
    generated_text = tokenizer.sequences_to_texts(generated_indices)

    return start_string + generated_text


def main():
    global data_size, vocab_size, char_to_ix, ix_to_char, tokenizer
    lines = []

    corpus, tokenizer, X, Y = load_dataset("dinos", mode="line_by_line", lowercase=True)

    # 5. Pad sequences
    X = pad_sequences(X, padding="post")
    Y = pad_sequences(Y, padding="post")

    # 6. Tensorify
    X = tf.constant(X, dtype=tf.int32)
    Y = tf.constant(Y, dtype=tf.int32)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Create model
    vocab_size = len(tokenizer.get_vocab())  # from your tokenizer
    embedding_dim = 64
    rnn_units = 128
    model = TFCharRNN(vocab_size, embedding_dim, rnn_units)

    # Compile
    model.compile(optimizer="adam", loss=loss_fn)

    # Define a custom callback for live sampling
    class TextSampler(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 2 == 0:  # Sample every 2 epochs
                print(f"\n--- Sampling after epoch: {epoch + 1} ---")
                for _ in range(3):  # Generate 3 samples
                    generated_text = generate_text(
                        model,
                        start_string="",
                        tokenizer=tokenizer,
                        max_length=50,
                        temperature=1.0,  # Can vary
                    )
                    last_dino_name = generated_text[0].upper() + generated_text[1:]
                    print(last_dino_name)
                # print("\n")

    # Train
    model.fit(X, Y, epochs=10, callbacks=[TextSampler()])

    print("Generating samples:")
    for _ in range(7):
        generated_text = generate_text(
            model,
            start_string="",
            tokenizer=tokenizer,
            max_length=50,
            temperature=1.0,  # Can vary
            # temperature=0.5,  # Can vary
        )
        last_dino_name = generated_text[0].upper() + generated_text[1:]
        print(last_dino_name)


if __name__ == "__main__":
    main()
