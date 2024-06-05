class ActivationFunction:
    def __init__(self):
        pass

    def activate(self, x: float) -> float:
        pass

    def derivative(self, x: float) -> float:
        pass


class ReLU(ActivationFunction):
    def activate(self, x):
        return max(0.0, x)

    def derivative(self, x):
        return 1.0 if x > 0.0 else 0.0


class Sigmoid(ActivationFunction):
    def activate(self, x):
        return 1.0 / (1.0 + pow(2.718281828459045, -x))

    def derivative(self, x):
        return self.activate(x) * (1 - self.activate(x))


class TanH(ActivationFunction):
    def activate(self, x):
        return 2.0 / (1.0 + pow(2.718281828459045, -2 * x)) - 1.0

    def derivative(self, x):
        return 1 - pow(self.activate(x), 2)
