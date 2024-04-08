from typing import List
from optimizers import *
from activation_functions import *
from initializers import *


class NeuralNetwork:
    def __init__(self, layers, activation_function, initializer, optimizer):
        self.layers: List[int] = layers.copy()
        self.neurons: List[List[float]] = []
        self.biases: List[List[float]] = []
        self.delta: List[List[float]] = []
        self.weights: List[List[List[float]]] = []
        self.weights_delta: List[List[List[float]]] = []

        self.activation_function = activation_function
        self.initializer = initializer
        self.optimizer = optimizer

        self.init_neurons()
        self.init_biases()
        self.init_weights()

    def init_neurons(self):
        for layer in self.layers:
            self.neurons.append([0.0] * layer)
            self.delta.append([0.0] * layer)

    def init_biases(self):
        for layer in self.layers:
            self.biases.append([self.initializer.initialize()] * layer)

    def init_weights(self):
        for i in range(len(self.layers)):
            self.weights.append(
                [
                    [self.initializer.initialize() for _ in range(self.layers[i - 1])]
                    for _ in range(self.layers[i])
                ]
            )
            self.weights_delta.append(
                [
                    [0.0 for _ in range(self.layers[i - 1])]
                    for _ in range(self.layers[i])
                ]
            )

    def feed_forward(self, inputs: List[float]):
        self.neurons[0] = inputs.copy()

        for i in range(1, len(self.layers)):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j] = self.biases[i][j]
                for k in range(len(self.neurons[i - 1])):
                    self.neurons[i][j] += self.neurons[i - 1][k] * self.weights[i][j][k]
                self.neurons[i][j] = self.activation_function.activate(
                    self.neurons[i][j]
                )

        return self.neurons[-1]

    def back_propagation(self, inputs: List[float], targets: List[float]):
        self.feed_forward(inputs)

        for i in range(self.layers[-1]):
            self.delta[-1][i] = self.neurons[-1][i] - targets[i]

        for i in range(len(self.layers) - 2, 0, -1):
            for j in range(self.layers[i]):
                error = 0.0
                for k in range(self.layers[i + 1]):
                    error += self.delta[i + 1][k] * self.weights[i + 1][k][j]
                self.delta[i][j] = error * self.activation_function.derivative(
                    self.neurons[i][j]
                )

        self.optimizer.update(self)


if __name__ == "__main__":
    pass
