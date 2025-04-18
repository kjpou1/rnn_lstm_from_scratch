# 🧠 Activation Functions (From Scratch)

This module defines core activation functions used in training neural networks **built manually using NumPy**. Each activation follows a common interface via an abstract base class.

---

## 📦 Structure

All activation classes inherit from:

```python
BaseActivation (abstract class)
```

Each activation must implement:

```python
@staticmethod
def forward(x: np.ndarray) -> np.ndarray:
    """Apply the activation function."""

@staticmethod
def backward(x: np.ndarray) -> np.ndarray:
    """Compute derivative of the activation w.r.t. input (∂activation/∂x)."""
```

| Method        | Purpose                                                  | Used In               |
|---------------|----------------------------------------------------------|------------------------|
| `forward(x)`  | Apply non-linearity to input                             | Forward pass (`a`)     |
| `backward(x)` | Compute local derivative of activation                   | Backward pass (`dz`)   |

---

## 🧮 Why Separate `forward` and `backward`?

Training a neural network involves two phases:

1. **Forward Propagation** – Compute activations for each layer  
   → e.g., `a = tanh(z)`
2. **Backward Propagation** – Compute gradients to update weights  
   → e.g., `dz = da * (1 - tanh²(z))`

To support manual backpropagation, each activation defines:
- `forward()` for computing activations
- `backward()` for computing the local gradient

---

### 🔄 Forward–Backward Caching (and Why We Use It)

In this project, each activation function computes its **derivative using the output from its forward pass**. That means during training:

```python
# Forward pass
a = SigmoidActivation.forward(z)

# Backward pass
dz = da * SigmoidActivation.backward(a)  # NOT passing z
```

This is a **conscious design decision** made for clarity and modularity.

> 💡 Rather than mixing activation logic with hidden state or class-level storage, we favor **explicit data flow**: forward results must be saved and passed to backward. It’s simple, transparent, and avoids surprises.

---

### 📚 Comparison to NNFS (Neural Networks from Scratch)

The book **NNFS** uses class-based activations that **store internal state** (like `self.output`) across the forward and backward phases:

```python
class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * self.output * (1 - self.output)
```

While that design supports clean encapsulation, it can obscure intermediate computations when debugging or learning.

In contrast, this project uses:
- Stateless `@staticmethod` functions  
- Clear data ownership — the caller is responsible for saving intermediate values  
- No hidden state or side-effects

This is in line with our educational goal:  
✅ *See exactly what’s flowing forward and backward — every step, every value.*

---

## ✅ Supported Activations

| Activation | Class Name         | Forward Function                  | Derivative (`backward()`)                    |
|------------|--------------------|-----------------------------------|----------------------------------------------|
| **Tanh**   | `TanhActivation`   | `tanh(x)`                         | `1 - tanh²(x)`                                |
| **Sigmoid**| `SigmoidActivation`| `1 / (1 + exp(-x))`               | `σ(x)(1 - σ(x))` where `σ` is sigmoid        |
| **Softmax**| `SoftmaxActivation`| `e^x / Σ e^x` (vector output)     | ∂𝑎ᵢ/∂𝑥ⱼ = Jacobian matrix (see below)        |
| **ReLU**   | `ReLUActivation`   | `max(0, x)`                       | `1 if x > 0 else 0`                           |

---

## ✨ Usage Example

```python
from activations.tanh import TanhActivation

z = np.array([...])  # pre-activation

# Forward pass
a = TanhActivation.forward(z)

# Backward pass (manual gradient computation)
dz = da * TanhActivation.backward(z)
```

---

## 🔬 Manual Derivatives vs. Autograd (Frameworks)

In this scratch implementation, **you must define both `forward()` and `backward()`**, because we're not using automatic differentiation.

### 🧠 Example: Sigmoid

```python
def forward(x):
    return 1 / (1 + np.exp(-x))

def backward(x):
    s = forward(x)
    return s * (1 - s)
```

---

## 🤖 What Do Frameworks Do?

Libraries like **TensorFlow** and **PyTorch**:

- Record all operations in a **computational graph**
- Use **automatic differentiation** (autograd)
- Handle chain rule and derivatives internally

### ✅ TensorFlow Example

```python
import tensorflow as tf

x = tf.Variable([[0.0, 1.0]])
with tf.GradientTape() as tape:
    y = tf.math.sigmoid(x)

dy_dx = tape.gradient(y, x)
print(dy_dx)  # no need to define backward()
```

---

## 🔍 Summary: Manual vs. Automatic Differentiation

| Feature                 | Scratch RNNs        | TensorFlow / PyTorch    |
|------------------------|---------------------|--------------------------|
| Activation `forward`   | ✅ Required          | ✅ Required               |
| Activation `backward`  | ✅ Required          | ❌ Handled by autograd    |
| Backpropagation Logic  | Manual chain rule    | Automatic gradient engine|
| Flexibility            | 🔧 Fully Customizable| ⚡ Highly Optimized       |

---
## 📚 Reference & Credit

The design of this activation module — with explicit `forward()` and `backward()` methods — was inspired by:

### 🧠 Neural Networks from Scratch in Python  
by *Harrison Kinsley (Sentdex)* and *Daniel Kukieła*

This book (and YouTube series) walks through neural networks at a truly foundational level, including:

- Manual implementation of activations (sigmoid, tanh, ReLU, etc.)
- Derivative math for backpropagation
- Forward + backward interface design patterns

If you're learning how to build neural nets from first principles, it’s an incredible resource.

- 🔗 [Official Site](https://nnfs.io)  
- 📺 [YouTube Playlist](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfSfqQuee6K8opKtZsh7sA9)  
- 💻 [GitHub Code](https://github.com/Sentdex/nnfs)

---

### 🎓 Coursera: Machine Learning Specialization

From *Andrew Ng's DeepLearning.AI* — this course introduces the theory and math behind many activation functions used in modern neural networks:

- Sigmoid, Tanh, and ReLU explained with derivatives  
- Chain rule applications in backpropagation  
- When to use each function (and why)

- 🧮 [Course: Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms)
- 🌍 [Coursera Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

