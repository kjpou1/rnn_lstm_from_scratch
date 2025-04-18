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
