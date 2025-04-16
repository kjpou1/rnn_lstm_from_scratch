# generate_lstm_expected.py
# -------------------------------------------------------------------------
# This script runs the LSTM forward pass on a fixed input configuration and
# prints out the final hidden state and output (softmax predictions) for use
# in unit tests.
#
# ✅ IMPORTANT: This must use the SAME random seed and dimensions as the unit test.
#    Otherwise, the generated values will differ, and test comparisons will fail.
#
# ➕ Usage:
#   Run this script to regenerate EXPECTED_A_FINAL and EXPECTED_Y_FINAL values
#   for use in `tests/lstm/test_lstm_forward.py`.
# -------------------------------------------------------------------------

import numpy as np

from src.lstm_model import initialize_lstm_parameters, lstm_forward

# 🔒 Set random seed for reproducibility — MUST match the unit test exactly
np.random.seed(42)

# 📐 Dimensions — must match the unittest config
n_x, n_a, n_y = 3, 5, 2  # input, hidden, and output sizes
m, T_x = 4, 3  # batch size and time steps

# 🧪 Generate synthetic test input
x = np.random.randn(n_x, m, T_x)
a0 = np.random.randn(n_a, m)
parameters = initialize_lstm_parameters(n_a, n_x, n_y)

# 🚀 Run forward pass
a, y, _ = lstm_forward(x, a0, parameters)

# 📦 Get final time step outputs
a_final = a[:, :, -1]
y_final = y[:, :, -1]


# 🧾 Pretty-print helper for pasting into test code
def format_array(arr, name):
    print(f"\n{name} = np.array([")
    for row in arr:
        formatted = ", ".join(f"{x:.7f}" for x in row)
        print(f"    [{formatted}],")
    print("])")


# 🖨️ Output final arrays
format_array(a_final, "EXPECTED_A_FINAL")
format_array(y_final, "EXPECTED_Y_FINAL")
