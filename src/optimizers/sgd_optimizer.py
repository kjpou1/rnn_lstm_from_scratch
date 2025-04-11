from .optimizer import Optimizer

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

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
        
        # Use parent apply_gradients
        self.apply_gradients(grads_and_vars)

        return parameters
