# ✅ RNN Unit Test Coverage

This directory contains a comprehensive suite of unit tests to verify the correctness of your **from-scratch RNN implementation using NumPy**. The tests validate forward and backward propagation, gradient shapes and values, and compare numerical outputs against Keras’ `SimpleRNN` for correctness.

---

## 🔬 Test Breakdown

| Test File                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `test_rnn_backwards.py`          | Tests the full backward pass through time. Verifies gradient shapes, values, and logs stats. |
| `test_rnn_backwards_vs_keras.py` | Compares gradients from the scratch RNN with those from a Keras RNN using `GradientTape`. Useful for validating correctness of backprop implementation. |
| `test_rnn_forward_vs_keras.py`   | Runs the scratch `rnn_forward()` vs a Keras RNN model. Compares raw logits (`z_t`) and softmax predictions for alignment. |
| `test_rnn_vs_keras.py`           | Validates a single `rnn_cell_step()` against Keras’ `SimpleRNNCell`. Checks `a_next`, logits, and softmax outputs for each batch. |
| `test_loss_curve.py`             | Validates that loss decreases over training iterations on synthetic data. Checks smoothed loss and gradient flow. |
| `test_sampling_behavior.py`      | Verifies text generation behavior. Checks text length, vocabulary validity, newline termination, and temperature sensitivity. |

---

## ⚙️ Reproducibility

- All tests set `np.random.seed()` and `tf.random.set_seed()` to ensure deterministic behavior.
- Scratch RNN and Keras models use identical parameter shapes and manually set weights for accurate comparisons.
- Tests assume standard input formats: tokenized character sequences, index-based input/output, and 1D time steps.

---

## 🧪 Running the Tests

To run the full test suite:

```bash
python -m unittest discover -s tests/rnn
```

To run a specific module:

```bash
python -m unittest tests.rnn.test_rnn_forward_vs_keras
```

To enable debug output, run with verbose mode:

```bash
python -m unittest -v tests.rnn.test_sampling_behavior
```

---

## 🧰 Developer Notes

- Activate your virtual environment first:

  ```bash
  source .venv/bin/activate
  ```

- Always run with the correct PYTHONPATH:

  ```bash
  PYTHONPATH=. python -m unittest ...
  ```

- All tests are **self-contained** — no external datasets or checkpoints are required.
- Most tests include logs for inspection (`Max diff`, logits, samples, gradient stats) to simplify debugging.

---

## 📌 Summary

| Area             | Covered |
|------------------|---------|
| RNN cell logic   | ✅       |
| Forward pass     | ✅       |
| Backpropagation  | ✅       |
| Gradient shapes  | ✅       |
| Keras parity     | ✅       |
| Sampling tests   | ✅       |
| Loss curve test  | ✅       |
