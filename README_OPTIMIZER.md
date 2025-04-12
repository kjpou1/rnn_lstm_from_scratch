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
- **α** is the learning rate
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

Handled by the `SGDOptimizer` class.

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

Choosing the right learning rate **(α)** and momentum **(β)** is critical:

| Parameter | Typical Range | Effect |
|:----------|:--------------|:-------|
| **Learning rate (α)** | 0.001 – 0.01 | Smaller values help prevent overshooting when momentum is high |
| **Momentum coefficient (β)** | 0.9 – 0.99 | Higher values smooth updates but can cause overshooting |

---

### ⚡ Practical Observations

- **If the learning rate is too large** (e.g., 0.1 or higher):
  - Momentum causes **overshooting** and unstable training.

- **Solution:**
  - Lower learning rate to 0.01 or 0.001.
  - Tune β carefully (typically start with 0.9).

- **If β is too high** (e.g., 0.99+):
  - Momentum builds too much and may "blow past" minima.

✅ In scratch experiments, **MomentumOptimizer** performed **poorly with α = 0.1**, but improved **significantly when α = 0.01** or smaller.

---

### 🏃‍♂️ Quick Guide

| Setup | Behavior |
|:------|:---------|
| **High α + High β** | 🚀 Overshoots! Model unstable |
| **Low α + High β**  | 🏃‍♂️ Smooth fast convergence |
| **Too Low α**       | 🐢 Very slow learning |

---

> ⚡ **Bottom line:**  
> **Momentum can accelerate convergence**, but **only if α and β are tuned together carefully**.

---

## 🚀 3. RMSProp Optimizer

### ✍️ Update Rule (Math)

Given:
- Running average of squared gradients **s**
- Parameters **θ**
- Gradients **∇θ J(θ)**

The **RMSProp** update is:

```
s := β * s + (1 - β) * (∇θ J(θ))²
θ := θ - α * ∇θ J(θ) / (√s + ε)
```

Where:
- **s** is the exponentially decaying average of squared gradients
- **β** is the decay rate (e.g., 0.9)
- **ε** is a small constant to prevent division by zero (e.g., 1e-8)

---

### 📚 In Context of Our RNN

```
s_dWaa := β * s_dWaa + (1 - β) * (dWaa)²
Waa    := Waa - α * dWaa / (√s_dWaa + ε)
```
(Similar for `Wax`, `Wya`, `ba`, and `by`.)

---

### ✅ Key Properties
| Property | Behavior |
|:---------|:---------|
| Adaptive Learning Rates | Adjusts step size per parameter |
| Reduces Oscillations | Especially for noisy or sparse gradients |
| Hyperparameter Sensitivity | Requires tuning β and α |

---

### 📉 When to Use
- **Training is noisy**
- **Gradients vary a lot**
- **Want automatic learning rate adaptation**

✅ In practice, RMSProp speeds up convergence compared to SGD or plain momentum.

---

## 🚀 4. Adam Optimizer

### ✍️ Update Rule (Math)

Adam combines **Momentum** + **RMSProp**:

1. **First moment estimate (mₜ):**

```
mₜ = β₁ * mₜ₋₁ + (1 - β₁) * ∇θ J(θ)
```

2. **Second moment estimate (vₜ):**

```
vₜ = β₂ * vₜ₋₁ + (1 - β₂) * (∇θ J(θ))²
```

3. **Bias correction:**

```
m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)
```

4. **Update parameters:**

```
θ := θ - α * m̂ₜ / (√v̂ₜ + ε)
```

Where:
- **α** = learning rate
- **β₁** = momentum term (first moment, e.g., 0.9)
- **β₂** = RMSProp term (second moment, e.g., 0.999)
- **ε** = small value to prevent division by zero

---

### 📚 In Context of Our RNN

```
Waa := Waa - α * m̂_dWaa / (√v̂_dWaa + ε)
```
(Similar updates for `Wax`, `Wya`, `ba`, and `by`.)

---

### ✅ Key Properties
| Property | Behavior |
|:---------|:---------|
| Combines Momentum + RMSProp | ✅ |
| Adaptive Learning Rate | ✅ |
| Bias Correction | ✅ |
| Very Popular Optimizer | ✅ |

---

### 📉 When to Use
- **Almost always** a strong default choice
- Robust to noisy gradients
- Works well without much hyperparameter tuning

✅ In our scratch experiments, Adam delivered **fast, stable convergence**.

---

# 📜 Optimizer Status

| Optimizer | Status |
|:----------|:-------|
| SGD (vanilla) | ✅ Implemented |
| Momentum | ✅ Implemented |
| RMSProp | ✅ Implemented |
| Adam | ✅ Implemented |

---

# 🛠️ Optimizer File Structure

```
src/
└── optimizers/
    ├── optimizer.py          # Base optimizer class
    ├── sgd_optimizer.py      # SGD optimizer
    ├── momentum_optimizer.py # Momentum optimizer
    ├── rmsprop_optimizer.py  # RMSProp optimizer
    ├── adam_optimizer.py     # Adam optimizer
```

---

> 💡 **Reminder:**  
> Our goal is to *learn by doing*, so every optimizer is implemented manually using **NumPy**, no TensorFlow or PyTorch optimizers.

---

## 🧠 Learn More
- [Coursera NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/1)
- [Backpropagation Through Time (BPTT)](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)
