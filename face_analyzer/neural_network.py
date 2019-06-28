import numpy as np


class NeuralNetwork:
    def __init__(self, num_hidden_neurons, activation_func, num_inputs):
        self._start_network_layers(num_hidden_neurons, num_inputs)
        self.activation_func = activation_func

    def _start_network_layers(self, num_hidden_neurons, num_input):
        self.weights = np.random.rand(num_hidden_neurons, num_input)
        self.bias = np.zeros((num_hidden_neurons, 1))
        self.bias = np.add(self.bias, 1)

    def calculate_first_layer_output(self, first_face, second_face):
        # resizing inputs
        first_face_copy = first_face.copy()
        first_face_copy = np.resize(first_face_copy, (len(first_face_copy), 1))
        second_face_copy = second_face.copy()
        second_face_copy = np.resize(second_face_copy, (len(first_face_copy), 1))
        input_matrix = np.append(first_face_copy, second_face_copy, axis=0)
        # multiplying weights with input
        result = np.dot(self.weights, input_matrix)
        print("shape of weights: " + str(self.weights.shape))
        print("shape of input_matrix: " + str(input_matrix.shape))
        print("shape of result: " + str(input_matrix.shape))
        print(" mult weight X input: " + str(result))
        # executing sigmoid function to each result
        result = np.resize(np.array([self.activation_func(x) for x in result]), result.shape)
        print(" activition over mult : " + str(result))
        # adding bias
        result = np.resize(np.array([value+self.bias[index, 0] for index, value in enumerate(result)]), result.shape)
        print(" now adding bias : " + str(result))
        return result
