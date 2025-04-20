# RNN-LSTM-from-Scratch

Welcome to **RNN-LSTM-from-Scratch** — a hands-on lab where we **build deep learning models from the ground up**, using only **NumPy**.

Learn how **recurrent neural networks** (RNNs) and **long short-term memory networks** (LSTMs) really work — no black boxes, no shortcuts.

---

## 📚 Table of Contents

- [RNN-LSTM-from-Scratch](#rnn-lstm-from-scratch)
  - [📚 Table of Contents](#-table-of-contents)
  - [🚀 Why This Project?](#-why-this-project)
  - [🎯 Project Philosophy](#-project-philosophy)
  - [🧠 What's Inside](#-whats-inside)
    - [✅ Core Components (NumPy)](#-core-components-numpy)
    - [🧠 Models (`src/models/`)](#-models-srcmodels)
    - [🧮 Activations (`src/activations/`)](#-activations-srcactivations)
    - [🏋️ Training Scripts](#️-training-scripts)
    - [⚙️ Optimizers (`src/optimizers/`)](#️-optimizers-srcoptimizers)
  - [🤖 Two Ways to Train](#-two-ways-to-train)
  - [📈 Features Overview](#-features-overview)
  - [🧠 Design Decisions](#-design-decisions)
    - [❗Logits instead of Softmax in the Forward Pass](#logits-instead-of-softmax-in-the-forward-pass)
  - [✅ Tests](#-tests)
    - [🔁 RNN Tests](#-rnn-tests)
    - [🧠 LSTM Tests](#-lstm-tests)
  - [📁 Project Layout](#-project-layout)
  - [📚 Datasets](#-datasets)
  - [🧪 Running the Code](#-running-the-code)
    - [🔹 Single-Example Training](#-single-example-training)
      - [CLI Arguments](#cli-arguments)
    - [🔹 Mini-Batch Training](#-mini-batch-training)
      - [CLI Arguments](#cli-arguments-1)
  - [✍️ Example Output](#️-example-output)
  - [🛠️ Coming Soon](#️-coming-soon)
  - [🧩 Built With](#-built-with)
  - [📬 Contributing](#-contributing)
  - [🎓 Learn More](#-learn-more)

---

## 🚀 Why This Project?
- Master the **nuts and bolts** of RNNs and LSTMs by building everything yourself.
- See exactly **how models learn** through **forward passes**, **backward passes**, and **optimizer updates**.
- Compare **pure NumPy training** vs **TensorFlow workflows** — side-by-side.
- Build, train, debug, and **generate text** with models you fully understand.

---

## 🎯 Project Philosophy

This project isn't just about repeating what's taught — it's about breaking things down, rebuilding them, and truly understanding **how neural networks tick**. We dig under the hood to understand the math, the matrix ops, and the mechanics that make RNNs and LSTMs work.

Our goal is to **learn by building**, not just by using. That means stepping away from the Keras high-level API and rebuilding each part from scratch — even if it’s harder, slower, and less elegant. The result? You walk away understanding not just *what* to do, but *why* it works.

> Think of this as a laboratory for neural networks — part engineering, part science fair, part curiosity project.

---

## 🧠 What's Inside

### ✅ Core Components (NumPy)
- `CharTokenizer`: Maps characters ↔️ indices (with OOV support)
- `data_prep.py`: Loads text and prepares training sequences
- `utils.py`: Utility functions for `softmax`, loss smoothing, padding, clipping
- `sampling.py`: Shared sampling logic (RNN, LSTM) with temperature scaling

---

### 🧠 Models (`src/models/`)
- `rnn_model.py`: Recurrent Neural Network (RNN)
  - Forward pass, backpropagation through time (BPTT), loss via logits
- `lstm_model.py`: Long Short-Term Memory (LSTM)
  - Forward pass, backward pass, modular activation support

---

### 🧮 Activations (`src/activations/`)
- `tanh.py`: Hyperbolic tangent with manual forward/backward
- `sigmoid.py`: Sigmoid activation with manual gradients
- `softmax.py`: Temperature-scaled softmax (vectorized + single-column)
- `base.py`: Abstract activation interface (for modular design)

📄 See [`README_ACTIVATION.md`](README_ACTIVATION.md) for full math, gradients, and usage notes.

---

### 🏋️ Training Scripts
- `scratch_char_level_rnn_model.py`: Single-example RNN training (manual)
- `scratch_char_level_rnn_model_batch.py`: Mini-batch RNN training (manual)
- `tf_char_level_rnn_model.py`: TensorFlow model (`model.fit`)
- `tf_char_rnn_manual_train.py`: TensorFlow with manual training loop
- `tf_char_rnn.py`: TensorFlow RNN class (`TFCharRNN`)

---

### ⚙️ Optimizers (`src/optimizers/`)
- `SGDOptimizer`: Basic stochastic gradient descent
- `MomentumOptimizer`: SGD with momentum
- `RMSPropOptimizer`: Adaptive learning with RMS decay
- `AdamOptimizer`: Adaptive Moment Estimation (Adam)

📄 Full docs: [`README_OPTIMIZER.md`](README_OPTIMIZER.md)

---

## 🤖 Two Ways to Train
| Mode                 | Framework        | Description |
|:---------------------|:-----------------|:------------|
| **From Scratch**      | NumPy             | Full manual RNN training with explicit forward/backward passes |
| **TensorFlow**        | TensorFlow (Keras) | High-level training for benchmarking and comparison |

---

✅ TensorFlow versions (in `tf_char_*` files) are **fully isolated** and **not used** in scratch training.  
✅ Compare the low-level and high-level approaches **side-by-side**.

---

## 📈 Features Overview

| Feature                   | NumPy (Scratch) | TensorFlow Version | Status |
|:---------------------------|:----------------|:-------------------|:------:|
| Character tokenizer         | ✅ | ✅ | Complete |
| Manual forward pass         | ✅ | ✅ | Complete |
| Manual backward pass (BPTT) | ✅ | ✅ | Complete |
| Mini-batching               | ✅ | ✅ | Complete |
| Clipping gradients          | ✅ | ✅ | Complete |
| Training with `model.fit`   | ❌ | ✅ | Complete |
| Manual training loop        | ✅ | ✅ | Complete |
| Sampling (temperature)      | ✅ | ✅ | Complete |
| LSTM cell                   | 🔜 | 🔜 | Coming Soon |
| Optimizers (SGD, RMSProp, Adam) | ✅ | ✅ | Complete |

---

## 🧠 Design Decisions

This project **intentionally deviates** from the Coursera implementation in key places to align better with deep learning best practices:

### ❗Logits instead of Softmax in the Forward Pass

Unlike the original course, we do **not apply softmax during the forward pass**. Instead, we return the raw output logits and defer softmax to the loss function or sampling step.

**Why?**

- ✅ This mirrors modern deep learning frameworks like TensorFlow and PyTorch, where loss functions (e.g. `categorical_crossentropy`) handle softmax internally via `from_logits=True`.
- ✅ It improves **numerical stability** and reduces unnecessary computation during training.
- ✅ It lets us compute a **vectorized softmax** once during loss calculation, instead of step-by-step during the forward pass.

This was a **conscious architectural decision** — not a shortcut.  
It helps us better debug gradients, align with frameworks, and prepare for deeper experiments like temperature sampling and attention mechanisms.

---

## ✅ Tests

### 🔁 RNN Tests

The RNN implementation is backed by a comprehensive test suite to ensure forward and backward logic are both mathematically correct and consistent with TensorFlow.

- ✅ Validates `rnn_cell_step()`, `rnn_forward()`, and `rnn_backward()` end-to-end
- 🔬 Compares gradients with Keras using `GradientTape`
- 📉 Confirms loss decreases over synthetic training loops
- 🧪 Tests sampling output: sequence length, vocab conformity, temperature effects

📄 See [`tests/rnn/README.md`](tests/rnn/README.md) for a full breakdown.  
All tests are deterministic, self-contained, and require **no external data**.

---

### 🧠 LSTM Tests

This repo includes a from-scratch LSTM implementation with detailed unit tests for:

- Forward/backward cell logic
- Gradient shape and value correctness
- Cross-validation against Keras `LSTMCell`

📄 Full breakdown in [`tests/lstm/README.md`](tests/lstm/README.md)

---

## 📁 Project Layout

- `src/` — Core training and model code (NumPy + TensorFlow)
  - `models/` — From-scratch model logic
    - `rnn_model.py`, `lstm_model.py`
  - `data_prep.py`, `tokenizer.py`, `utils.py`, `sampling.py`
  - `scratch_*.py` (from-scratch trainers)
  - `tf_*.py` (TensorFlow trainers)
  - `activations/` — Hand-coded activation functions 🔬
  - `optimizers/` — Custom SGD, Momentum, RMSProp, Adam 💡

- `tests/` — Unit tests for RNN, LSTM, optimizers, and sampling
- `data/` — Datasets like `dinos.txt`, `shakespeare.txt`
- `README_ACTIVATION.md`, `README_OPTIMIZER.md` — Docs for custom components

---

## 📚 Datasets
- `dinos.txt` — Dinosaur names
- `shakespeare.txt` — Shakespeare plays

*Want to train on your own text?*  
Just drop a `.txt` file into the `data/` folder — you’re ready to go!

---

## 🧪 Running the Code

This project includes two scratch-built NumPy training scripts:

- 🧬 `scratch_char_level_rnn_model.py`: single-example RNN training
- 🧪 `scratch_char_level_rnn_model_batch.py`: mini-batch RNN training

> ✅ Before running either script, be sure to set:
```bash
export PYTHONPATH=.
```

---

### 🔹 Single-Example Training

```bash
python -m src.scratch_char_level_rnn_model

or

PYTHONPATH=. python -m src.scratch_char_level_rnn_model
```

#### CLI Arguments

| Argument            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `--dataset`         | Dataset name (e.g. `dinos` → `data/dinos.txt`)                              |
| `--iterations`      | Number of training iterations                                               |
| `--learning_rate`   | Learning rate for gradient descent                                          |
| `--optimizer`       | Optimizer type: `sgd`, `momentum`, `rms`, or `adam`                         |
| `--temperature`     | Sampling temperature: <br>`<1` = more deterministic, `>1` = more creative   |
| `--hidden_size`     | Number of RNN hidden units                                                  |
| `--sample_every`    | Print sampled text every N iterations                                       |
| `--seq_length`      | Maximum length of generated samples                                         |
| `--clip_value`      | Gradient clipping threshold                                                 |

---

### 🔹 Mini-Batch Training

```bash
python -m src.scratch_char_level_rnn_model_batch

or 

PYTHONPATH=. python -m src.scratch_char_level_rnn_model_batch
```

#### CLI Arguments

| Argument            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `--dataset`         | Dataset name (e.g. `dinos`)                                                 |
| `--epochs`          | Number of training epochs (full data passes)                                |
| `--batch_size`      | Size of mini-batches                                                        |
| `--learning_rate`   | Learning rate                                                               |
| `--optimizer`       | Optimizer type: `sgd`, `momentum`, `rms`, or `adam`                         |
| `--temperature`     | Sampling temperature                                                        |
| `--hidden_size`     | Hidden layer size                                                           |
| `--seq_length`      | Length of generated sequences                                               |
| `--clip_value`      | Max allowed gradient norm (clipping)                                       |
| `--deterministic`   | Set this flag for deterministic shuffling (reproducibility)                |

> The mini-batch script uses **line-by-line training** and applies `pad_sequences()` to handle variable input lengths.


---

## ✍️ Example Output
After training on dinosaur names:
```
--- Generating samples:
Tonisaurus

Opandeniaurus

Onosaurus

Tonisaurus

Yptops

Eunthus

Yosaurus
```

---

## 🛠️ Coming Soon
| Feature | Status |
|:--------|:-------|
| LSTM Cell from scratch | 🔜 In Progress |
| GRU Cell from scratch | 🔜 In Progress |
| Model checkpointing | ⏳ |
| Attention mechanism exploration | 🧠 Future |

---

## 🧩 Built With
- Python 3.10+
- NumPy (pure scratch mode)
- TensorFlow (comparison mode)

---

## 📬 Contributing
Pull Requests welcome!  
This is a **learning-first lab** for anyone who wants to truly understand RNNs and LSTMs — no magic, no shortcuts. 🧪

---

## 🎓 Learn More

To deepen your understanding of the building blocks behind RNNs and LSTMs:

- 📘 [Neural Networks from Scratch (Book)](https://nnfs.io/) – Great resource for understanding manual backprop, activations, and optimizers from first principles.
- 🧠 [Coursera NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/1) – Excellent series covering RNNs, GRUs, LSTMs, and BPTT.
- 🔄 [Backpropagation Through Time (BPTT)](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time) – Understand how gradients flow across time steps.
- 🎯 [Hyperparameter Tuning & Optimization](https://www.coursera.org/learn/deep-neural-network) – For training stability and performance improvements.

> Want to understand how `tanh`, `sigmoid`, and `softmax` really work?  
> 📂 Check out [`README_ACTIVATION.md`](./README_ACTIVATION.md) for hand-coded formulas, derivatives, and examples — no magic, just math.

---

Happy building 🔁  

Let's make it learn 🦖 ➡️ 📝  
🦖 Train it. 🔁 Backprop it. ✍️ Sample it.  
Build deep, learn deeper.

Become one with the gradients.
