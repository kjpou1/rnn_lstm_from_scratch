import tensorflow as tf


class TFCharRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.layers.SimpleRNN(
            rnn_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs)

        # ⬇️ Just pass states if given (for inference), else let RNN auto-handle
        x, state = self.rnn(x, initial_state=states, training=training)
        x = self.dense(x)
        if return_state:
            return x, state
        return x
