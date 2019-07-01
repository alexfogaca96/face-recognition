from math import exp


class ActivationFunctions:
    @staticmethod
    def sigmoid_function(x):
        return  1/(1 + exp(-x))

    
    #@staticmethod
    #def derivative_sigmoid(x):
    #    sig_func = ActivationFunctions.sigmoid_function
    #    return sig_func(x) * (1 - sig_func(x))