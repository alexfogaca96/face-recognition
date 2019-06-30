from math import exp


class ActivationFunctions:
    @staticmethod
    def sigmoid_function(x):
<<<<<<< HEAD
        return  1/(1 + math.exp(-x))

    
    #@staticmethod
    #def derivative_sigmoid(x):
    #    sig_func = ActivationFunctions.sigmoid_function
    #    return sig_func(x) * (1 - sig_func(x))
=======
        return 1 / 1 + exp(-x)
>>>>>>> 8891c50cae05c9b9db49d14e6d92de7992a5a221
