from math import exp


class ActivationFunctions:
    @staticmethod
    def sigmoid_function(x):
        return 1 / 1 + exp(-x)
