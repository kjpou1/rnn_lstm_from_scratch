# src/optimizers/sgd_optimizer.py

from .optimizer import Optimizer

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, parameters, gradients):
        """
        Traditional scratch RNN parameter update.

        Args:
            parameters (dict): Model parameters.
            gradients (dict): Gradients.
        
        Returns:
            Updated parameters dict.
        """
        grads_and_vars = [(gradients["d" + key], parameters[key]) for key in parameters.keys()]
        self.apply_gradients(grads_and_vars)
        return parameters
