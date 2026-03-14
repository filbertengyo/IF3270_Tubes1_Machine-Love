import numpy as np
from abc import ABC, abstractmethod

class PerceptronOptimizer(ABC):
    '''Abstract class for perceptron optimizers'''

    def __init__(self, weights: np.ndarray):
        self.weights = weights.copy()

    @abstractmethod
    def optimize(self, gradients: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class GradientDescentOptimizer(PerceptronOptimizer):
    '''Perceptron optimizer utilizing standard gradient descent'''

    def __init__(self, weights: np.ndarray, learning_rate: float):
        self.weights = weights
        self.learning_rate = learning_rate

    def optimize(self, gradients: np.ndarray) -> np.ndarray:
        self.weights -= gradients * self.learning_rate
        return self.weights


class AdamOptimizer(PerceptronOptimizer):
    '''Perceptron optimizer utilizer ADAM method'''

    EPSILON = 1e-8

    def __init__(self, weights: np.ndarray, learning_rate: float, momentum_gain: float, rms_gain: float):
        self.weights = weights
        self.momentum = np.zeros_like(weights)
        self.rms = np.zeros_like(weights)

        self.learning_rate = learning_rate
        self.momentum_gain = momentum_gain
        self.rms_gain = rms_gain

        self.steps = 1

    def optimize(self, gradients: np.ndarray) -> np.ndarray:
        uncorrected_momentum = self.momentum_gain * self.momentum + (1 - self.momentum_gain) * gradients
        uncorrected_rms = self.rms_gain * self.rms + (1 - self.rms_gain) * (gradients ** 2)
        
        self.momentum = uncorrected_momentum / (1 - self.momentum_gain ** self.steps)
        self.rms = uncorrected_rms / (1 - self.rms_gain ** self.steps)
        self.steps += 1

        self.weights -= self.momentum / (np.sqrt(self.rms) + self.EPSILON) * self.learning_rate
        return self.weights