# RNN-LSTM-from-Scratch

This project implements character-level **RNNs and LSTMs** **from scratch** using only **NumPy**, alongside **TensorFlow versions** for comparison.

It’s a hands-on deep dive into **how RNNs and LSTMs really work — from first principles.**

---

## 🚀 Goals
- Understand and **implement RNNs and LSTMs** from scratch
- Train models to generate text character-by-character
- Explore **forward and backward passes** (including BPTT)
- Apply **gradient clipping** to stabilize training
- Build **custom optimizers** (SGD, RMSProp, Adam)
- Compare **scratch NumPy** vs **TensorFlow** workflows side-by-side

---

## 🧠 What's Inside

### ✅ Core Features
- `CharTokenizer`: Maps characters ↔️ indices (with OOV support)
- `data_prep.py`: Load raw text and create training sequences
- `rnn_model.py`: RNN core logic (forward, backward, sampling, clipping)

- **Training Scripts**:
  - `scratch_char_level_rnn_model.py`: Single example iteration training (NumPy)
  - `scratch_char_level_rnn_batch_train.py`: Mini-batch training (NumPy)
  - `tf_char_level_rnn_model.py`: TensorFlow model (uses `model.fit`)
  - `tf_char_rnn_manual_train.py`: TensorFlow **manual training loop** (custom batches)
  - `tf_char_rnn.py`: TensorFlow model class (`TFCharRNN`)

- `utils.py`: Helper functions: softmax, loss smoothing, random seeds, etc.
- `optimizers.py`: Custom optimizer implementations (SGD, RMSProp, Adam)

---

## 🤖 Two Modes of Training
| Mode                 | Framework        | Description |
|:---------------------|:-----------------|:------------|
| **Scratch RNN**       | NumPy             | Full manual implementation (educational focus) |
| **TensorFlow RNN**    | TensorFlow (Keras) | Used for comparison (manual batch training + fit training) |

---

✅ TensorFlow versions (in `tf_char_*` files) are **fully isolated** and **not used** in scratch RNN training.

✅ This lets you **compare side-by-side**:
- *From-scratch NumPy workflow* vs *TensorFlow/Keras best practices*  
- *Manual gradient updates* vs *automatic optimizers*

---

## 📈 Features Table

| Feature            | NumPy (Scratch) | TensorFlow Version | Status |
|:-------------------|:----------------|:-------------------|:------:|
| Character tokenizer | ✅ | ✅ | Complete |
| Manual forward pass | ✅ | ✅ | Complete |
| Manual backward pass (BPTT) | ✅ | ✅ | Complete |
| Mini-batching        | ✅ | ✅ | Complete |
| Clipping gradients   | ✅ | ✅ | Complete |
| Training with model.fit | ❌ | ✅ | Complete |
| Manual training loop | ✅ | ✅ | Complete |
| Sampling (temperature) | ✅ | ✅ | Complete |
| LSTM cell            | 🔜 | 🔜 | Coming Soon |
| Optimizers (SGD, RMSProp, Adam) | ✅ | ✅ | Complete |

---

## 📂 Project Structure
```
rnn-lstm-from-scratch/
├── data/                         # Text corpora (e.g., dinos, Shakespeare)
├── src/
│   ├── tokenizer.py              # CharTokenizer
│   ├── data_prep.py              # Dataset loading and preparation
│   ├── rnn_model.py              # Scratch RNN core (forward, backward)
│   ├── scratch_char_level_rnn_model.py  # Iteration-based scratch trainer
│   ├── scratch_char_level_rnn_batch_train.py  # Batch-based scratch trainer
│   ├── tf_char_rnn.py            # TensorFlow RNN Model Class
│   ├── tf_char_level_rnn_model.py # TensorFlow model with .fit
│   ├── tf_char_rnn_manual_train.py # TensorFlow model with manual training
│   ├── utils.py                  # Softmax, smoothing, random seed, etc.
│   ├── optimizers.py             # Custom optimizer implementations
└── README.md
```

---

## 📚 Datasets
- `dinos.txt` — Dinosaur names
- `shakespeare.txt` — Shakespeare plays

You can add your own character-level corpus easily (just drop a `.txt` file into `data/`).

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
- NumPy only (pure scratch)
- No TensorFlow or PyTorch for scratch models

---

## 📬 Contributing
Pull Requests welcome!  
We are building a true **from-scratch RNN-LSTM lab** 🧪

---

## 🧠 Learn More
- [Coursera NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/1)
- [Backpropagation Through Time (BPTT)](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)

---

Happy building 🔁  
Let's make it learn 🦖 ➡️ 📝
