from abc import ABC, abstractmethod

import numpy as np


class BaseActivation(ABC):
    """
    Abstract base class for activation functions.
    Enforces a consistent interface for forward and backward operations.
    """

    @staticmethod
    @abstractmethod
    def forward(x: np.ndarray) -> np.ndarray:
        """Compute activation output"""
        pass

    @staticmethod
    @abstractmethod
    def backward(x: np.ndarray) -> np.ndarray:
        """Compute derivative w.r.t. input"""
        pass
