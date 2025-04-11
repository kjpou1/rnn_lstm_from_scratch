class Optimizer:
    def __init__(self, learning_rate=0.01):
        """
        Base Optimizer class.

        Args:
            learning_rate (float): Step size for parameter updates.
        """
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """
        Updates parameters given gradients.

        This is the **scratch-style** method:
        - Accepts dictionaries of parameters and gradients.
        - Internally prepares (gradient, parameter) pairs.
        - Calls `apply_gradients` to perform the update.

        Args:
            parameters (dict): Dictionary of current parameters.
            gradients (dict): Dictionary of gradients.

        Returns:
            dict: Updated parameters.
        """
        grads_and_vars = []
        for param_name in parameters:
            grad_name = "d" + param_name  # convention: "dWaa", "dWax", etc.
            grads_and_vars.append((gradients[grad_name], parameters[param_name]))

        self.apply_gradients(grads_and_vars)

        return parameters

    def apply_gradients(self, grads_and_vars):
        """
        Applies gradients to parameters.

        This is the **TensorFlow-style** method:
        - Accepts a list of (gradient, parameter) tuples.
        - Applies the update rule.

        Args:
            grads_and_vars (list): List of (gradient, parameter) tuples.
        """
        for grad, param in grads_and_vars:
            param -= self.learning_rate * grad
