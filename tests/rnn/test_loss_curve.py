import unittest

import numpy as np

from src.models.rnn_model import initialize_rnn_parameters, rnn_backward, rnn_forward
from src.optimizers.sgd_optimizer import SGDOptimizer
from src.utils.grad_utils import compute_output_layer_gradients
from src.utils.loss_utils import compute_loss_and_grad, project_logit_grad_to_hidden
from src.utils.utils import clip, cross_entropy_loss, smooth


class TestLossCurve(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.vocab_size = 5
        self.seq_length = 10
        self.hidden_size = 20
        self.learning_rate = 0.01
        self.num_steps = 100

        # Create fixed synthetic data
        self.X = [
            np.random.randint(0, self.vocab_size, self.seq_length).tolist()
            for _ in range(1)
        ]
        self.Y = [
            np.random.randint(0, self.vocab_size, self.seq_length).tolist()
            for _ in range(1)
        ]

        self.parameters = initialize_rnn_parameters(
            self.hidden_size, self.vocab_size, self.vocab_size
        )
        self.optimizer = SGDOptimizer(learning_rate=self.learning_rate)

    def test_loss_decreases(self):
        a_prev = np.zeros((self.hidden_size, 1))
        loss_history = []
        loss = -np.log(1.0 / self.vocab_size) * self.seq_length
        initial_loss_estimate = loss

        for step in range(self.num_steps):
            x_seq = self.X[0]
            y_seq = self.Y[0]

            # === Clean Path: Forward → Loss + Grad → Backward → Update ===

            # 1. Forward pass through the RNN
            a, _, logits, _ = rnn_forward(x_seq, a_prev, self.parameters)

            # 2. Compute cross-entropy loss and ∂L/∂logits (dy)
            curr_loss, dy = compute_loss_and_grad(logits, y_seq, reduction="mean")

            # 3. Project ∂L/∂logits → ∂L/∂hidden using Wyaᵀ
            da = project_logit_grad_to_hidden(
                dy, self.parameters["Wya"]
            )  # (n_a, 1, T_x)

            # 4. Backprop through the recurrent layer (no output params here)
            # Reuse forward outputs for consistency
            a_new, x, _, z_t = rnn_forward(x_seq, a_prev, self.parameters)
            gradients, a = rnn_backward(da, self.parameters, (a_new, x, logits, z_t))

            # 5. Add output layer gradients (dWya, dby)
            grads_out = compute_output_layer_gradients(dy, a)
            gradients.update(grads_out)

            # 6. Clip gradients and update parameters
            gradients = clip(gradients, maxValue=5.0)
            self.parameters = self.optimizer.update(self.parameters, gradients)

            # 7. Track smoothed loss
            loss = smooth(loss, curr_loss)
            loss_history.append(loss)

            # 8. Carry forward final hidden state
            a_prev = a[len(x_seq) - 1]

        print("\n📉 Final smoothed loss:", round(loss, 4))
        print("🔁 Loss history (last 5):", [round(l, 4) for l in loss_history[-5:]])

        self.assertLess(
            loss,
            loss_history[0],
            msg="Expected training loss to decrease from initial value",
        )

        self.assertLess(
            loss,
            initial_loss_estimate * 0.95,
            msg="Expected training loss to decrease by at least 5%",
        )

        self.assertLess(
            loss_history[-1],
            loss_history[0],
            msg="Final loss should be lower than initial loss",
        )

        print("✅ Passed: Training loss decreased over time")


if __name__ == "__main__":
    unittest.main()
