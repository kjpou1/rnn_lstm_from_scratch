from .base_optimizer import BaseOptimizer


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This is the simplest optimization algorithm which updates parameters
    by moving them in the direction of the negative gradient scaled by the learning rate.

    Inherits the `update()` method from BaseOptimizer which:
    - Extracts gradients from a parameter dictionary.
    - Passes them to this `apply_gradients()` method for the update logic.

    Attributes:
        learning_rate (float): Step size used to scale gradients during updates.
    """

    def apply_gradients(self, grads_and_vars):
        """
        Applies vanilla SGD updates to parameters.

        Args:
            grads_and_vars (list of tuples): Each tuple contains
                (gradient, parameter) to be updated.
        """
        for grad, param in grads_and_vars:
            # Subtract gradient scaled by learning rate (SGD rule)
            param -= self.learning_rate * grad
