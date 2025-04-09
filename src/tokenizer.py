# rnn-lstm-from-scratch/src/tokenizer.py

"""
Character-level Tokenizer with support for:
- Vocabulary generation
- Character to index mapping
- Index to character mapping
- OOV (out-of-vocabulary) token handling
- Encoding text to index sequences and decoding sequences to text

This implementation is inspired by Keras's Tokenizer but built from scratch.
"""


class CharTokenizer:
    def __init__(self, oov_token="<UNK>"):
        """
        Initialize the tokenizer.

        Args:
            oov_token (str): Token to represent out-of-vocabulary characters.
        """
        self.oov_token = oov_token
        self.char_to_ix = {}
        self.ix_to_char = {}
        self.vocab = []
        self.oov_index = None

    def fit(self, corpus):
        if isinstance(corpus, list):
            corpus = " ".join(corpus)

        chars = sorted(set(corpus))

        special_tokens = []
        if self.oov_token:
            special_tokens.append(self.oov_token)

        chars = special_tokens + chars

        self.vocab = chars
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for ch, i in self.char_to_ix.items()}

        self.oov_index = self.char_to_ix[self.oov_token] if self.oov_token else None
        # self.start_index = self.char_to_ix["<START>"]

    def texts_to_sequences(self, corpus):
        """
        Convert input text to a sequence of character indices.

        Args:
            corpus (str or list of str): Input text or list of sentences.

        Returns:
            list: List of character indices.
        """
        if isinstance(corpus, list):
            corpus = " ".join(corpus)
        return [self.char_to_ix.get(ch, self.oov_index) for ch in corpus]

    def sequences_to_texts(self, sequence):
        """
        Convert a sequence of indices back to the corresponding character string.

        Args:
            sequence (list): List of character indices.

        Returns:
            str: Decoded text string.
        """
        return "".join([self.ix_to_char.get(idx, self.oov_token) for idx in sequence])

    def get_vocab(self):
        """
        Return the vocabulary list.

        Returns:
            list: List of characters in the vocabulary.
        """
        return self.vocab

    def vocab_size(self):
        """
        Return the size of the vocabulary.

        Returns:
            int: Vocabulary size.
        """
        return len(self.vocab)


# Example usage
if __name__ == "__main__":
    corpus = ["hello world", "hola mundo"]
    tokenizer = CharTokenizer(oov_token="<UNK>")
    tokenizer.fit(corpus)

    print("Vocabulary:", tokenizer.get_vocab())
    encoded = tokenizer.texts_to_sequences("hello")
    print("Encoded:", encoded)
    decoded = tokenizer.sequences_to_texts(encoded)
    print("Decoded:", decoded)
