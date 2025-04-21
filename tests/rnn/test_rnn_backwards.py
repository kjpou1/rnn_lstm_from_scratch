import unittest

import numpy as np

from src.models.rnn_model import initialize_rnn_parameters, rnn_backward, rnn_forward
from src.utils.grad_utils import compute_output_layer_gradients
from src.utils.loss_utils import compute_loss_and_grad, project_logit_grad_to_hidden


class TestRNNBackwards(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)

        # Model config
        self.n_a = 5
        self.m = 1  # single example
        self.T_x = 6
        self.vocab_size = 3
        self.n_x = self.vocab_size
        self.n_y = self.vocab_size  # softmax over vocab

        # Create tokenized input and target sequences
        self.X = np.random.randint(0, self.vocab_size, size=self.T_x).tolist()
        self.Y = np.random.randint(0, self.vocab_size, size=self.T_x).tolist()

        # Initial hidden state
        self.a0 = np.random.randn(self.n_a, self.m).astype(np.float32)

        # Initialize model weights
        self.parameters = initialize_rnn_parameters(
            self.n_a, self.vocab_size, self.n_y, seed=3
        )

        # Run forward pass
        self.a, self.x_cache, self.logits, self.z_t = rnn_forward(
            self.X, self.a0, self.parameters
        )

        # Store cache for backward
        self.cache = (self.a, self.x_cache, self.logits, self.z_t)

        _, self.dy = compute_loss_and_grad(self.logits, self.Y, reduction="sum")
        self.da = project_logit_grad_to_hidden(
            self.dy, self.parameters["Wya"]
        )  # (n_a, 1, T_x)

    def test_backward_shapes(self):
        grads, _ = rnn_backward(self.da, self.parameters, self.cache)
        grads_out = compute_output_layer_gradients(self.dy, self.a)
        grads.update(grads_out)

        self.assertEqual(grads["dWax"].shape, self.parameters["Wax"].shape)
        self.assertEqual(grads["dWaa"].shape, self.parameters["Waa"].shape)
        self.assertEqual(grads["dWya"].shape, self.parameters["Wya"].shape)
        self.assertEqual(grads["dba"].shape, self.parameters["ba"].shape)
        self.assertEqual(grads["dby"].shape, self.parameters["by"].shape)

        print("\n✅ Passed: rnn_backward produces correctly shaped gradients")

    def test_backward_gradient_stats(self):
        grads, _ = rnn_backward(self.da, self.parameters, self.cache)
        grads_out = compute_output_layer_gradients(self.dy, self.a)
        grads.update(grads_out)

        print("\n📊 Gradient stats:")
        for name, grad in grads.items():
            print(
                f"{name}: mean={np.mean(grad):.6f}, std={np.std(grad):.6f}, max={np.max(np.abs(grad)):.6f}"
            )

        print("✅ Passed: rnn_backward gradient stats logged")


if __name__ == "__main__":
    unittest.main()
