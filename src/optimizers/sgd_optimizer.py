class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """
        Update parameters dictionary (scratch RNN style) using apply_gradients internally.

        Args:
            parameters (dict): Dictionary of parameters.
            gradients (dict): Dictionary of gradients.

        Returns:
            dict: Updated parameters.
        """
        # Build grads_and_vars list
        grads_and_vars = []
        for key in parameters.keys():
            grad = gradients["d" + key]
            param = parameters[key]
            grads_and_vars.append((grad, param))
        
        # Apply gradients
        self.apply_gradients(grads_and_vars)

        return parameters

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients from list of (gradient, parameter) tuples (TensorFlow-style).
        """
        for grad, param in grads_and_vars:
            param -= self.learning_rate * grad
