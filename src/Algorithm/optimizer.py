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

    def __init__(self, weights: np.ndarray, learning_rate: float, l1_strength: float, l2_strength: float):
        super().__init__(weights)
        self.learning_rate = learning_rate
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

    def optimize(self, gradients: np.ndarray) -> np.ndarray:
        l1_grad = np.sign(self.weights) * self.l1_strength
        l2_grad = 2 * self.weights * self.l2_strength
        regularized_gradients = gradients + l1_grad + l2_grad

        self.weights -= regularized_gradients * self.learning_rate
        return self.weights


class AdamOptimizer(PerceptronOptimizer):
    '''Perceptron optimizer utilizer ADAM method'''

    EPSILON = 1e-8

    def __init__(self, weights: np.ndarray, learning_rate: float, momentum_gain: float, rms_gain: float, l1_strength: float, l2_strength: float):
        super().__init__(weights)
        self.momentum = np.zeros_like(weights)
        self.rms = np.zeros_like(weights)

        self.learning_rate = learning_rate
        self.momentum_gain = momentum_gain
        self.rms_gain = rms_gain

        self.l1_strength = l1_strength
        self.l2_strength = l2_strength

        self.steps = 1

    def optimize(self, gradients: np.ndarray) -> np.ndarray:
        l1_grad = np.sign(self.weights) * self.l1_strength
        l2_grad = 2 * self.weights * self.l2_strength
        regularized_gradients = gradients + l1_grad + l2_grad

        self.momentum = self.momentum_gain * self.momentum + (1 - self.momentum_gain) * regularized_gradients
        self.rms = self.rms_gain * self.rms + (1 - self.rms_gain) * (regularized_gradients ** 2)
        
        corrected_momentum = self.momentum / (1 - self.momentum_gain ** self.steps)
        corrected_rms = self.rms / (1 - self.rms_gain ** self.steps)
        self.steps += 1

        self.weights -= corrected_momentum / (np.sqrt(corrected_rms) + self.EPSILON) * self.learning_rate
        return self.weights