# 📈 Optimizer Documentation

This file documents the **optimizers** used in the `rnn-lstm-from-scratch` project.

We focus on **implementing optimizers manually** to better understand how parameter updates work during training.

[Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)

---

## 🚀 1. Stochastic Gradient Descent (SGD)

### ✍️ Update Rule (Math)

Given:
- Parameter `θ` (such as weights or biases)
- Loss function `J(θ)`
- Gradient `∇θ J(θ)`

The **SGD update** is:

```
θ := θ - α * ∇θ J(θ)
```

Where:
- `α` is the **learning rate**
- `∇θ J(θ)` is the gradient of the loss w.r.t. the parameter

---

### 📚 In Context of Our RNN

If parameters are:
- `Waa`, `Wax`, `Wya`, `ba`, `by`

And gradients are:
- `dWaa`, `dWax`, `dWya`, `dba`, `dby`

Then the updates are:

```
Waa := Waa - α * dWaa
Wax := Wax - α * dWax
Wya := Wya - α * dWya
ba  := ba  - α * dba
by  := by  - α * dby
```

---

### 🧩 Python Pseudocode

```python
for param_name in parameters:
    parameters[param_name] -= learning_rate * gradients["d" + param_name]
```

In our code, this is handled by the `SGDOptimizer` class.

---

### ✅ Key Properties
| Property | Behavior |
|:---------|:---------|
| Simplicity | Very easy to implement |
| Memory Usage | Very low (no extra state needed) |
| Convergence | Can be slow if learning rate is not tuned |
| Instability | Sensitive to learning rate, no momentum |

---

### 📉 When to Use
- **Simple problems**
- **Small datasets**
- When learning rate is carefully tuned manually

---

## 📜 Upcoming Optimizers

| Optimizer | Status |
|:----------|:-------|
| SGD (vanilla) | ✅ Implemented |
| Momentum | 🔜 In Progress |
| RMSProp | 🔜 Planned |
| Adam | 🔜 Planned |

---

✅ As we add more optimizers like **Momentum**, **RMSProp**, and **Adam**, they will be documented here with:
- Update equations
- Behavior summary
- Code examples

---

# 🛠️ File Structure (Coming Soon)

```
src/
└── optimizers/
    ├── optimizer_base.py    # Base optimizer class
    ├── sgd_optimizer.py     # SGD optimizer
    ├── momentum_optimizer.py # Momentum optimizer (planned)
    ├── rmsprop_optimizer.py # RMSProp optimizer (planned)
    └── adam_optimizer.py    # Adam optimizer (planned)
```

---

> 💡 **Reminder:**  
> Our goal is to *learn by doing*, so every optimizer is implemented manually with **NumPy**, no external libraries like TensorFlow or PyTorch optimizers.

---

## 🧠 Learn More
- [Coursera NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/1)
- [Backpropagation Through Time (BPTT)](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)
