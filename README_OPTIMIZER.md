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

The **SGD update rule** is:

```
θ := θ - α ∇θ J(θ)
```

Where:
- **α** is the **learning rate**
- **∇θ J(θ)** is the **gradient** of the loss function with respect to the parameter θ

---

### 📚 In Context of Our RNN

If parameters are:
- **Waa**, **Wax**, **Wya**, **ba**, **by**

And gradients are:
- **dWaa**, **dWax**, **dWya**, **dba**, **dby**

Then the updates are:

```
Waa := Waa - α dWaa
Wax := Wax - α dWax
Wya := Wya - α dWya
ba  := ba  - α dba
by  := by  - α dby
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

| Property        | Behavior                                |
|:----------------|:----------------------------------------|
| Simplicity      | Very easy to implement                  |
| Memory Usage    | Very low (no extra state needed)         |
| Convergence     | Can be slow if learning rate not tuned  |
| Instability     | Sensitive to learning rate, no momentum |

---

### 📉 When to Use

- **Simple problems**
- **Small datasets**
- When learning rate is carefully tuned manually

---

## 🚀 2. Momentum Optimizer

### ✍️ Update Rule (Math)

Momentum adds a velocity term **v** to smooth updates:

```
v := β v - α ∇θ J(θ)
θ := θ + v
```

Where:
- **β** is the **momentum coefficient** (typically 0.9)
- **v** is the **velocity** (running sum of past gradients)
- **α** is the **learning rate**
- **∇θ J(θ)** is the gradient

---

### 📚 In Context of Our RNN

The parameter updates now consider *previous gradients*:

1. Update velocity:

```
v_dWaa = β v_dWaa - α dWaa
v_dWax = β v_dWax - α dWax
...
```

2. Update parameters:

```
Waa := Waa + v_dWaa
Wax := Wax + v_dWax
...
```

---

### 🧩 Python Pseudocode

```python
for grad, param in grads_and_vars:
    v = momentum * v - learning_rate * grad
    param += v
```

In our code, this is handled by the `MomentumOptimizer` class.

---

### ✅ Key Properties

| Property        | Behavior                                |
|:----------------|:----------------------------------------|
| Smoother updates | Less oscillation compared to SGD       |
| Faster convergence | Can speed up training significantly |
| Requires tuning | Need to choose **α** and **β** carefully |

---

### 📉 When to Use

- Training is slow or oscillatory
- Need to escape local minima
- Common when learning simple RNNs or small LSTMs

---

## 📜 Optimizer Status

| Optimizer  | Status         |
|:-----------|:---------------|
| SGD        | ✅ Implemented |
| Momentum   | ✅ Implemented |
| RMSProp    | 🔜 Planned     |
| Adam       | 🔜 Planned     |

---

✅ As we add **RMSProp** and **Adam**, we will document them here with:
- Update equations
- Python pseudocode
- Practical usage notes

---

# 🛠️ Optimizer Code Layout

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
> All optimizers are implemented manually with **NumPy**, no external libraries like TensorFlow or PyTorch.

---

## 🧠 Learn More

- [Coursera NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models/home/week/1)
- [Backpropagation Through Time (BPTT)](https://www.coursera.org/learn/nlp-sequence-models/lecture/bc7ED/backpropagation-through-time)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network)

---
