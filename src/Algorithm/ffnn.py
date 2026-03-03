import numpy as np

class FFNN:
    "Feedforward Neural Network implementation."
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1]) * 0.01
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    # silahkan lanjutin