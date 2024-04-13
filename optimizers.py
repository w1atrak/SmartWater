import math


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, neural_network):
        pass


class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []
        self.v = []

    def update(self, neural_network):
        self.t += 1
        if not self.m:
            self.m = [
                [
                    [0.0 for _ in range(len(neural_network.neurons[i - 1]))]
                    for _ in range(len(neural_network.neurons[i]))
                ]
                for i in range(len(neural_network.weights))
            ]
            self.v = [
                [
                    [0.0 for _ in range(len(neural_network.neurons[i - 1]))]
                    for _ in range(len(neural_network.neurons[i]))
                ]
                for i in range(len(neural_network.weights))
            ]

        for i in range(len(neural_network.weights) - 1, 0, -1):
            for j in range(len(neural_network.neurons[i])):
                for k in range(len(neural_network.neurons[i - 1])):
                    g = -neural_network.delta[i][j] * neural_network.neurons[i - 1][k]

                    self.m[i][j][k] = (
                        self.beta1 * self.m[i][j][k] + (1 - self.beta1) * g
                    )
                    self.v[i][j][k] = (
                        self.beta2 * self.v[i][j][k] + (1 - self.beta2) * g * g
                    )
                    m_hat = self.m[i][j][k] / (1 - math.pow(self.beta1, self.t))
                    v_hat = self.v[i][j][k] / (1 - math.pow(self.beta2, self.t))

                    neural_network.weights[i][j][k] += (
                        self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
                    )


class GradientDescentOptimizer(Optimizer):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, neural_network):
        for i in range(len(neural_network.weights) - 1, 0, -1):
            for j in range(len(neural_network.neurons[i])):
                for k in range(len(neural_network.neurons[i - 1])):
                    neural_network.weights[i][j][k] -= (
                        self.learning_rate
                        * neural_network.delta[i][j]
                        * neural_network.neurons[i - 1][k]
                    )
