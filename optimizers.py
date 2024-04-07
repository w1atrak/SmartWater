class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, neural_network):
        pass


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0

    def update(self, neural_network):
        if not self.m:
            for i in range(1, len(neural_network.layers)):
                self.m.append(
                    [
                        [0.0 for _ in range(neural_network.layers[i - 1])]
                        for _ in range(neural_network.layers[i])
                    ]
                )
                self.v.append(
                    [
                        [0.0 for _ in range(neural_network.layers[i - 1])]
                        for _ in range(neural_network.layers[i])
                    ]
                )

        self.t += 1
        for i in range(1, len(neural_network.layers)):
            for j in range(neural_network.layers[i]):
                for k in range(neural_network.layers[i - 1]):
                    self.m[i - 1][j][k] = (
                        self.beta1 * self.m[i - 1][j][k]
                        + (1 - self.beta1)
                        * neural_network.delta[i][j]
                        * neural_network.neurons[i - 1][k]
                    )
                    self.v[i - 1][j][k] = (
                        self.beta2 * self.v[i - 1][j][k]
                        + (1 - self.beta2)
                        * (
                            neural_network.delta[i][j]
                            * neural_network.neurons[i - 1][k]
                        )
                        ** 2
                    )
                    m_hat = self.m[i - 1][j][k] / (1 - self.beta1**self.t)
                    v_hat = self.v[i - 1][j][k] / (1 - self.beta2**self.t)
                    neural_network.weights[i][j][k] -= (
                        self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)
                    )
                self.m[i - 1][j] = [0.0 for _ in range(neural_network.layers[i - 1])]
                self.v[i - 1][j] = [0.0 for _ in range(neural_network.layers[i - 1])]
