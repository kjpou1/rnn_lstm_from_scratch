# 📈 Optimizer Documentation

This file documents the **optimizers** used in the `rnn-lstm-from-scratch` project.

We focus on **implementing optimizers manually** to better understand how parameter updates work during training.

[Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)

---

## 🚀 1. Stochastic Gradient Descent (SGD)

### ✍️ Update Rule (Math)

Given:
- Parameter **θ** (such as weights or biases)
- Loss function **J(θ)**
- Gradient **∇θ J(θ)**

The **SGD update** is:

```
θ := θ - α * ∇θ J(θ)
```

Where:
- **α** is the **learning rate**
- **∇θ J(θ)** is the gradient of the loss w.r.t. the parameter.

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

## 🚀 2. Momentum Optimizer

### ✍️ Update Rule (Math)

Given:
- Velocity vector **v**
- Parameters **θ**
- Gradients **∇θ J(θ)**

The **Momentum** update is:

```
v := β * v - α * ∇θ J(θ)
θ := θ + v
```

Where:
- **β** is the momentum coefficient (e.g., 0.9)
- **α** is the learning rate

---

### 📚 In Context of Our RNN

```
v_dWaa := β * v_dWaa - α * dWaa
Waa    := Waa + v_dWaa
```
(Similar updates for `Wax`, `Wya`, `ba`, and `by`.)

---

### 🧠 Practical Observations from Training

**Choosing the right learning rate (α) and momentum (β) is critical:**

| Parameter | Typical Range | Effect |
|:----------|:--------------|:-------|
| **Learning rate (α)** | 0.001 – 0.01 | Smaller values help prevent overshooting when momentum is high |
| **Momentum coefficient (β)** | 0.9 – 0.99 | Higher values smooth updates but can also cause overshooting |

---

### ⚡ Practical Observations

- **If the learning rate is too large** (e.g., 0.1 or higher):
  - The momentum effect can cause **overshooting**.
  - Model oscillates wildly or diverges (loss increases).

- **Solution:**
  - **Lower the learning rate** (e.g., 0.01 or 0.001) when using momentum.
  - **Tune β carefully** — start with 0.9 and adjust slightly if needed.

- **If β is too high** (e.g., 0.99+):
  - Momentum builds up too much and may "blow past" the minimum.
  - Training becomes unstable unless learning rate is very small.

✅ In our scratch experiments, **MomentumOptimizer** performed poorly with **α = 0.1**, but **improved significantly** when we reduced **α** to **0.01** or even **0.001**.

---

### 🏃‍♂️ Quick Guide

| Setup | Behavior |
|:------|:---------|
| **High α + High β** | 🚀 Overshoots! Model unstable |
| **Low α + High β**  | 🏃‍♂️ Smooth fast convergence |
| **Too Low α**       | 🐢 Very slow learning |

---

> ⚡ **Bottom line:**  
> **Momentum can speed up convergence** and **smooth training**, but **only if the learning rate and momentum are carefully tuned together.**

---

## 📜 Current Optimizer Status

| Optimizer | Status |
|:----------|:-------|
| SGD (vanilla) | ✅ Implemented |
| Momentum | ✅ Implemented |
| RMSProp | 🔜 Planned |
| Adam | 🔜 Planned |

---

✅ As we add more optimizers like **RMSProp** and **Adam**, they will be documented here with:
- Update equations
- Behavior summary
- Code examples

---

# 🛠️ Optimizer File Structure

```
src/
└── optimizers/
    ├── optimizer.py          # Base optimizer class
    ├── sgd_optimizer.py      # SGD optimizer
    ├── momentum_optimizer.py # Momentum optimizer
    ├── rmsprop_optimizer.py  # (planned)
    └── adam_optimizer.py     # (planned)
```

---

> 💡 **Reminder:**  
> Our goal is to *learn by doing*, so every optimizer is implemented manually with **NumPy**, no TensorFlow or PyTorch optimizers.

---

## 🧠 Learn More
- [Coursera NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/1)
- [Backpropagation Through Time (BPTT)](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)
