from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers.
    Only `apply_gradients` needs to be overridden.
    `update` provides the standard interface.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, parameters, gradients):
        """
        Standardized parameter update using apply_gradients().
        """
        grads_and_vars = [
            (gradients["d" + key], parameters[key]) for key in parameters.keys()
        ]
        self.apply_gradients(grads_and_vars)
        return parameters

    @abstractmethod
    def apply_gradients(self, grads_and_vars):
        """
        Core optimizer logic goes here.
        Must be implemented in subclasses.
        """
        pass
