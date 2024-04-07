import random


class Initializer:
    def initialize(self):
        pass


class RandomInitializer(Initializer):
    def initialize(self):
        return random.uniform(-0.5, 0.5)
