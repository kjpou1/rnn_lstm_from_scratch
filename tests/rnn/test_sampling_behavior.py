import unittest

import numpy as np

from src.models.rnn_model import initialize_rnn_parameters
from src.sampling import generate_text
from src.tokenizer import CharTokenizer


class TestSamplingBehavior(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

        self.tokenizer = CharTokenizer(oov_token=None)
        self.tokenizer.fit(["abcde\n"])
        self.vocab = list(self.tokenizer.char_to_ix.keys())

        self.vocab_size = self.tokenizer.vocab_size()
        self.n_a = 10

        self.parameters = initialize_rnn_parameters(
            self.n_a, self.vocab_size, self.vocab_size, seed=42
        )

    def test_generated_text_length(self):
        max_length = 30
        text = generate_text(
            self.parameters,
            self.tokenizer,
            start_string="a",
            temperature=1.0,
            max_length=max_length,
            seed=1,
        )
        print("\n🔢 Generated Text (Length Check):", repr(text))
        self.assertLessEqual(
            len(text),
            max_length + 1,
            msg="Generated text exceeds max_length + 1 (accounting for start_string)",
        )

    def test_all_characters_in_vocab(self):
        text = generate_text(
            self.parameters,
            self.tokenizer,
            start_string="",
            temperature=1.0,
            max_length=50,
            seed=2,
        )
        print("\n🔤 Generated Text (Char Check):", repr(text))
        for char in text:
            self.assertIn(
                char,
                self.vocab,
                msg=f"Character '{char}' not in known vocabulary",
            )

    def test_generation_ends_with_newline(self):
        text = generate_text(
            self.parameters,
            self.tokenizer,
            start_string="",
            temperature=1.0,
            max_length=100,
            seed=3,
        )
        print("\n🔚 Generated Text (Newline Check):", repr(text))
        self.assertTrue(
            text.endswith("\n") or len(text) == 100,
            msg="Text should end with newline or reach max length",
        )

    def test_temperature_effect(self):
        text1 = generate_text(
            self.parameters,
            self.tokenizer,
            temperature=0.5,
            max_length=30,
            seed=42,
        )
        text2 = generate_text(
            self.parameters,
            self.tokenizer,
            temperature=1.5,
            max_length=30,
            seed=99,
        )

        print("\n🌡 Temp 0.5 Sample:", repr(text1))
        print("🌡 Temp 1.5 Sample:", repr(text2))

        self.assertNotEqual(
            text1,
            text2,
            "Temperature should affect sampling output",
        )


if __name__ == "__main__":
    unittest.main()
