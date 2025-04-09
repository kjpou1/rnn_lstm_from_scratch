class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """Directly update parameters. Traditional SGD update."""
        raise NotImplementedError

    def apply_gradients(self, grads_and_vars):
        """
        More flexible: accepts a list of (gradient, parameter) tuples.

        Mimics TensorFlow's optimizer.apply_gradients.
        """
        for grad, param in grads_and_vars:
            param -= self.learning_rate * grad


# # Scratch version
# parameters = optimizer.update(parameters, gradients)

# # More general version (future)
# grads_and_vars = [(gradients["dWaa"], parameters["Waa"]), (gradients["dWax"], parameters["Wax"]), ...]
# optimizer.apply_gradients(grads_and_vars)