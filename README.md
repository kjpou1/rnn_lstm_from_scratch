# RNN-LSTM-from-Scratch

Welcome to **RNN-LSTM-from-Scratch** — a hands-on lab where we **build deep learning models from the ground up**, using only **NumPy**.

Learn how **recurrent neural networks** (RNNs) and **long short-term memory networks** (LSTMs) really work — no black boxes, no shortcuts.

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

### ✅ Core Features
- `CharTokenizer`: Maps characters ↔️ indices (with OOV support)
- `data_prep.py`: Load text data and create training sequences
- `rnn_model.py`: RNN core logic (forward, backward, sampling, gradient clipping)

- **Training Scripts**:
  - `scratch_char_level_rnn_model.py`: Single example training (NumPy)
  - `scratch_char_level_rnn_batch_train.py`: Mini-batch training (NumPy)
  - `tf_char_level_rnn_model.py`: TensorFlow model (`model.fit` API)
  - `tf_char_rnn_manual_train.py`: TensorFlow model with manual training loop
  - `tf_char_rnn.py`: TensorFlow model class (`TFCharRNN`)

- `utils.py`: Helper functions: softmax, loss smoothing, random seeds
- `optimizers.py`: Custom-built SGD, RMSProp, and Adam optimizers

✅ See detailed optimizer docs [**here**](README_OPTIMIZER.md).

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

## ✅ RNN Tests

The RNN modules are backed by a comprehensive set of unit tests to ensure correctness in both forward and backward passes.

- 🔁 Validates `rnn_cell_step()`, `rnn_forward()`  and `rnn_backward()` against Keras
- 🎯 Compares gradients with `GradientTape` for numerical alignment
- 📉 Confirms training loss decreases over time
- 🧪 Checks sampling behavior, sequence length, and vocabulary coverage

📄 See [`tests/rnn/README.md`](tests/rnn/README.md) for the full suite. All tests are deterministic and self-contained.

---


## ✅ LSTM Tests

This repo includes a full from-scratch LSTM implementation with detailed unit tests (forward, backward, gradient check, Keras comparison).

📄 See [`tests/lstm/README.md`](tests/lstm/README.md) for full test breakdown.

---

## 📂 Project Structure
```
rnn-lstm-from-scratch/
├── data/
│   ├── images/                   # Images for visualizations
│   ├── dinos.txt                  # Dinosaur name corpus
│   └── shakespeare.txt            # Shakespeare plays corpus
│
├── src/
│   ├── optimizers/                # Custom optimizers
│   │   ├── __init__.py
│   │   ├── adam_optimizer.py
│   │   ├── momentum_optimizer.py
│   │   ├── optimizer.py           # Base optimizer class
│   │   ├── rmsprop_optimizer.py
│   │   └── sgd_optimizer.py
│   │
│   ├── __init__.py
│   ├── char_level_rnn_model.py    # Character-level RNN (NumPy)
│   ├── data_prep.py               # Dataset loading and preparation
│   ├── rnn_model.py               # Scratch RNN core (forward, backward)
│   ├── scratch_char_level_rnn_model_batch.py  # Scratch trainer (mini-batch)
│   ├── scratch_char_level_rnn_model.py        # Scratch trainer (single example)
│   ├── text_dataset.py            # Text dataset utilities
│   ├── tf_char_level_rnn_model.py # TensorFlow model with .fit
│   ├── tf_char_rnn_manual_train.py # TensorFlow manual training loop
│   ├── tf_char_rnn.py             # TensorFlow RNN model class
│   ├── tokenizer.py               # CharTokenizer
│   └── utils.py                   # Helper functions (softmax, loss smoothing, etc.)
│
├── tests/
│   ├── optimizers/
│   │   ├── test_adam_optimizer.py
│   │   ├── test_momentum_optimizer.py
│   │   ├── test_rmsprop_optimizer.py
│   │   └── test_sgd_optimizer.py
│
├── .gitignore
├── LICENSE
├── README.md
├── README_OPTIMIZER.md            # Optimizer-specific documentation
├── requirements.txt
```

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
Generated: "Brontosaurus"
Generated: "Stegoceratops"
Generated: "Trodonax"
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
