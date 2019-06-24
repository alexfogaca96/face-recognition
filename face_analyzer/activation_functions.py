import math
class ActivationFunctions():
    @staticmethod
    def sigmoid_function(x):
        return  (1/ 1 + math.exp(-x))