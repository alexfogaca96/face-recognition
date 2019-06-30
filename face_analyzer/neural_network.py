import numpy as np
import matplotlib.pyplot as plt
from enumerators.match_type import MatchType
class NeuralNetwork():
    def __init__(self , num_hidden_neurons, activation_func , num_inputs, num_output_neurons , starting_learn_rate):
        self.activation_func = activation_func
        self.learn_rate = starting_learn_rate
        self._start_network_layers(num_hidden_neurons, num_inputs , num_output_neurons)

    def _start_network_layers(self , num_hidden_neurons , num_input, num_output_neurons):
        self.i_h_weigths = np.random.rand(num_hidden_neurons, num_input)
        self.h_o_weigths = np.random.rand(num_output_neurons , num_hidden_neurons)  
        self.h_bias = np.zeros((num_hidden_neurons,1))
        self.h_bias = np.add(self.h_bias , 1)
        self.o_bias = np.zeros((num_output_neurons,1))
        self.o_bias = np.add(self.o_bias , 1)
        self.medium_error = np.array([0.1])
        self.medium_training_error = np.array([])

    def execute_training(self, training_validation_database):
        max_num_epocas = 200
        k = 0 
        traning_db_index = 0 
        while k < max_num_epocas and np.median(self.medium_error) > 0.0001:
            if(k == 0 ):
                self.medium_error = np.array([])
            if(traning_db_index == training_validation_database.training_db_size):
                traning_db_index = 0 
            traning_db = training_validation_database.get_training_database(traning_db_index)
            for data in traning_db:
                self.train(data)
            self.medium_error = np.append( self.medium_error , np.median(self.medium_training_error))
            self.medium_training_error  = np.array([])
            k = k +1
            print("actual epoch: " + str(k) )
        print("média de erros: " + str(np.median(self.medium_error)))
        print("erro da ultima época: " + str(self.medium_error[k-1]))
        for data in training_validation_database.validation_db:
            predict = self.get_output(data)
            correct = 0 
            if predict.value == data.match_type.value:
                correct += 1 
        print(len(training_validation_database.validation_db))
        print("taxa de acerto na base de validação foi de: " +str(correct/len(training_validation_database.validation_db)))

        x_index = np.arange(0,self.medium_error.size)
        plt.plot(x_index ,self.medium_error , 'ro')
        plt.show()

    def get_output(self, data ):
        first_face = data.face_one
        second_face = data.face_two
        #resizing inputs
        expected_result = self.get_expected_matrix_from_match_type(data.match_type)
        input_matrix = self.formatting_inputs(first_face , second_face)
        hidden_layer_output = self.calc_output_to_next_layer(self.i_h_weigths , input_matrix , self.h_bias)
        final_output = self.calc_output_to_next_layer(self.h_o_weigths , hidden_layer_output , self.o_bias)
        if(final_output[0] > final_output[1]):
            return MatchType.MATCH
        else:
            return MatchType.MISMATCH

    def train(self, data ):
        first_face = data.face_one
        second_face = data.face_two
        #resizing inputs
        expected_result = self.get_expected_matrix_from_match_type(data.match_type)
        input_matrix = self.formatting_inputs(first_face , second_face)
        #multiplying weights with input
        hidden_layer_output = self.calc_output_to_next_layer(self.i_h_weigths , input_matrix , self.h_bias)
        final_output = self.calc_output_to_next_layer(self.h_o_weigths , hidden_layer_output , self.o_bias)
        error_matrix = self.calc_error(final_output , expected_result)
        self.medium_training_error = np.append(self.medium_training_error ,np.sum(self.calculate_half_square_error(final_output , expected_result)))
        update_weigths ,update_bias= self.get_update_matrix(final_output , error_matrix , hidden_layer_output )
        self.h_o_weigths = np.add(self.h_o_weigths , update_weigths)
        self.o_bias =  np.add(self.o_bias,update_bias)
        error_matrix = np.dot(np.transpose(self.h_o_weigths) , error_matrix)
        update_weigths , update_bias = self.get_update_matrix(hidden_layer_output , error_matrix , input_matrix  )
        self.i_h_weigths  = np.add(self.i_h_weigths,update_weigths)
        self.h_bias  = np.add(self.h_bias,update_bias)
        
        '''
        error_matrix = self.calc_error(final_output , expected_result)
        derivative_func = lambda x : x * ( 1 - x)
        transpose_hidden_output = np.transpose(hidden_layer_output)
        gradient = derivative_func(final_output)
        gradient = np.multiply(gradient, error_matrix)
        gradient = np.multiply(gradient,self.learn_rate)
        print("gradiente: " + str(gradient))
        print("transpose hidden_output :" + str(transpose_hidden_output))
        update_matrix = np.dot(gradient , transpose_hidden_output)
        return update_matrix
        '''
    def get_update_matrix(self ,last_layer_output , error_matrix , previous_output  ):
        #print("error matrix:" + str(error_matrix))
        derivative_func = lambda x : x * ( 1 - x)
        t_previous_output = np.transpose(previous_output)
        gradient = derivative_func(last_layer_output)
        gradient = np.multiply(gradient, error_matrix)
        gradient = np.multiply(gradient,self.learn_rate)
        #print("gradiente: " + str(gradient))
        #print("transpose hidden_output :" + str(t_previous_output))
        update_matrix = np.dot(gradient , t_previous_output)
        #print("UPDATE MATRIX: " + str(update_matrix))
        return update_matrix , gradient

    def formatting_inputs(self , first_face , second_face):
        first_face_copy = first_face.copy()
        first_face_copy = np.resize(first_face_copy, (len(first_face_copy), 1))
        second_face_copy = second_face.copy()
        second_face_copy = np.resize(second_face_copy , (len(first_face_copy) , 1))
        input_matrix =np.append(first_face_copy, second_face_copy , axis=0)
        return input_matrix

    def calc_output_to_next_layer(self , weights , inputs , bias ):
        result =np.dot(weights, inputs)
        #print("multiplying for hidden to output weights:" + str(result))
        result = np.resize(np.array([ value+bias[index,0] for index, value in enumerate(result)]) , result.shape )
        #print(" now adding bias : " + str(result))
        result = np.resize(np.array([self.activation_func(x) for x in result]) , result.shape )
        #print(" activition : " + str(result))
        return result
    
    def calc_error(self , output , expected_result):
        #print("Expected result: "+str(expected_result))
        error = np.zeros((2,1))
        error = np.subtract(expected_result , output )
        #error = np.power(error , 2)
        #error = np.divide(error , 2 )
        #print ("Error matrix is equals to: " + str(error))
        return error

    def get_expected_matrix_from_match_type(self , match_type ):
        if match_type == MatchType.MATCH:
            return np.array([[1],[0]])
        else:
            return np.array([[0],[1]])

    def calculate_half_square_error(self , output , expected_result):
        error = np.zeros((2,1))
        error = np.subtract(expected_result , output )
        error = np.power(error , 2)
        error = np.divide(error , 2 )
        return error
