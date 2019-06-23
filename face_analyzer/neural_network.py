import numpy as np
class NeuralNetwork():
    def __init__(self , num_hidden_neurons, activation_func):
        self._start_network_layers(num_hidden_neurons)
        self.activation_func = activation_func

    def _start_network_layers(self , num_hidden_neurons):
        self.weigths = np.random.rand(2,num_hidden_neurons)  

    def calculate_first_layer_output(first_face , second_face):
        np.append(first_face , second_face , axis = 2)
            