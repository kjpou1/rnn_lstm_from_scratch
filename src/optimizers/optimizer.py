# src/optimizers/optimizer.py

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """
        Update parameters dict using gradients.

        Should call apply_gradients internally.
        """
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients from list of (gradient, parameter) tuples.

        TensorFlow-like behavior.
        """
        for grad, param in grads_and_vars:
            param -= self.learning_rate * grad
