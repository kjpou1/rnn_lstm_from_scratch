## ✅ LSTM Unit Test Coverage

This project includes a comprehensive suite of unit tests to validate the from-scratch LSTM implementation. Tests cover both forward and backward passes, gate-level computations, numerical gradient checks, and equivalence with Keras outputs.

### 🔬 Test Breakdown

| Test File                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `test_lstm_forward.py`            | Tests full-sequence LSTM forward pass — output shapes, final timestep values |
| `test_lstm_cell_step.py`          | Validates a single forward step of the LSTM cell                            |
| `test_lstm_step_backward.py`      | Backpropagation for one LSTM timestep — verifies all gate gradients         |
| `test_lstm_backwards.py`          | Full-sequence LSTM backward pass — accumulates gradients across time        |
| `test_lstm_backwards_vs_keras.py` | Compares analytical gradients with Keras `.gradient()` results              |
| `test_lstm_vs_keras.py`           | Confirms forward outputs match Keras LSTMCell + Dense model (with softmax)  |
| `test_numerical_grad_check.py`    | Numerically checks gradients (`dWf`, `dbf`, `dx`) against `lstm_backwards`  |
| `generate_lstm_expected.py`       | Script to regenerate test fixture data (`EXPECTED_A_FINAL`, `EXPECTED_Y_FINAL`) |

### ⚙️ Notes on Reproducibility

- **Some tests use `np.random.seed(42)`** for reproducibility — especially `test_lstm_forward.py` and `generate_lstm_expected.py`.
- Other tests (e.g. numerical gradient checks or backward validations) may use different seeds or rely on `np.random.seed(...)` explicitly inside the test method.
- Consistency between forward and backward test values is only guaranteed if the **same seed, shapes, and parameter initializers** are reused.
