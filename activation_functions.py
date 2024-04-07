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
