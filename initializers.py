import random
import csv


class Initializer:
    def initialize(self):
        pass


class RandomInitializer(Initializer):

    def initialize(self):
        return random.uniform(-0.5, 0.5)


class FileInitializer(Initializer):
    weights = []

    def __init__(self, suffix):
        try:
            with open(f"weights{suffix}.csv", "r") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    for v in list(map(float, row)):
                        self.weights.append(v)
            print("File loaded")
        except FileNotFoundError:
            print("File not found")
            pass

    def initialize(self, randomized=True):
        if randomized:
            return 0.3  # random.uniform(-0.5, 0.5)
        else:
            return self.weights.pop(0) if self.weights else random.uniform(-0.5, 0.5)
