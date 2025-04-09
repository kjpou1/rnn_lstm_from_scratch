from optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, parameters, gradients):
        """Traditional parameter dict update (used in scratch RNN)."""
        for key in parameters.keys():
            parameters[key] -= self.learning_rate * gradients["d" + key]
        return parameters

    # apply_gradients already works from base Optimizer
