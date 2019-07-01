import numpy as np
import matplotlib.pyplot as plt
from enumerators.match_type import MatchType
from saved_variables import SavedVariables
from pathlib import Path
import datetime

class NeuralNetwork():
    results_path = Path("results")
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
        

    def execute_training(self, training_validation_database):
        max_num_epocas = 500
        k = 0 
        validation_interval = 10
        self.medium_error = np.array([0.1])
        self.medium_training_error = np.array([])
        validation_error_list = np.array([])
        min_validation_error = float('inf')
        consecutive_deteriorate = 0

        while k < max_num_epocas and np.median(self.medium_error) > 0.0001 and consecutive_deteriorate < 5:
            if(k == 0 ):
                self.medium_error = np.array([])
            traning_db = training_validation_database.get_all_training_data()
            for data in traning_db:
                self.train(data)
            self.medium_error = np.append( self.medium_error , np.median(self.medium_training_error))
            self.medium_training_error  = np.array([])
            #a cada x epocas faz uma validação para verificar overfitting
            if k % validation_interval == 0:
                validation_db = training_validation_database.validation_db
                validation_error =self.calc_average_error_for_data_validation(validation_db)
                validation_error_list= np.append(validation_error_list ,validation_error)
                if min_validation_error > validation_error:
                    min_validation_error = validation_error
                    self.save()
                ## para ser considerado uma piora tem que ultrapassar: min_validation_error + deteriorate_limiter 
                else: 
                    consecutive_deteriorate += 1

            k = k +1
            print("actual epoch: " + str(k) )
        self.recover()
        self.generate_log(training_validation_database.validation_db  ,validation_error_list , validation_interval , k)


    def get_output(self, data ):
        first_face = data.face_one
        second_face = data.face_two
        final_output = self.run_feed_forward(first_face , second_face)
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
        update_weigths ,update_bias= self.get_update_matrix_l(final_output , error_matrix , hidden_layer_output )
        self.h_o_weigths = np.add(self.h_o_weigths , update_weigths)
        self.o_bias =  np.add(self.o_bias,update_bias)
        error_matrix = np.dot(np.transpose(self.h_o_weigths) , error_matrix)
        update_weigths , update_bias = self.get_update_matrix(hidden_layer_output , error_matrix , input_matrix  )
        self.i_h_weigths  = np.add(self.i_h_weigths,update_weigths)
        self.h_bias  = np.add(self.h_bias,update_bias)
        
    def get_update_matrix(self ,last_layer_output , error_matrix , previous_output  ):
        #print("error matrix:" + str(error_matrix))
        t_previous_output = np.transpose(previous_output)
        #derivative of sigmoid
        gradient = last_layer_output * (1 - last_layer_output)
        gradient = np.multiply(gradient, error_matrix)
        gradient = np.multiply(gradient,self.learn_rate)
        #print("gradiente: " + str(gradient))
        #print("transpose hidden_output :" + str(t_previous_output))
        update_matrix = np.dot(gradient , t_previous_output)
        #print("UPDATE MATRIX: " + str(update_matrix))
        return update_matrix , gradient
    
    def get_update_matrix_l(self ,last_layer_output , error_matrix , previous_output  ):
        #print("error matrix:" + str(error_matrix))
        t_previous_output = np.transpose(previous_output)
        #derivative of sigmoid
        gradient = last_layer_output 
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

    def run_feed_forward(self , first_face , second_face):
        input_matrix = self.formatting_inputs(first_face , second_face)
        hidden_layer_output = self.calc_output_to_next_layer(self.i_h_weigths , input_matrix , self.h_bias)
        final_output = self.calc_output_to_next_layer(self.h_o_weigths , hidden_layer_output , self.o_bias)
        return final_output

    def calc_average_error_for_data_validation(self , validation_db):
        errors = np.array([])
        for data in validation_db:
            first_face = data.face_one
            second_face = data.face_two
            #resizing inputs
            expected_result = self.get_expected_matrix_from_match_type(data.match_type)
            output = self.run_feed_forward(first_face , second_face)
            actual_errors = self.calculate_half_square_error( output , expected_result )
            errors = np.append(errors , actual_errors)
        avg_error =  np.average(errors)
        return avg_error

    def generate_log(self, validation_db,validation_error_list , validation_interval,k):
        traning_x = np.array([validation_interval*index for index , error in enumerate(validation_error_list) ])
        print("média de erros: " + str(np.median(self.medium_error)))
        print("erro da ultima época: " + str(self.medium_error[k-1]))
        print("taxa de aprendizado final: " + str(self.learn_rate))
        correct = 0
        for data in validation_db:
            predict = self.get_output(data) 
            if predict.value == data.match_type.value:
                correct += 1 
        print("taxa de acerto na base de validação foi de: " +str(correct/len(validation_db)))

        x_index = np.arange(0,self.medium_error.size)
        plt.plot(x_index ,self.medium_error  , color = 'skyblue' , label = "training error")
        plt.plot(traning_x , validation_error_list , color = 'olive' , label = "validation_error")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel("Error rate")
        plt.ylabel("Number of epochs")
        file_path = NeuralNetwork.results_path / ("Neural_Network_results_" +str(datetime.datetime.now()) + ".pdf")
        plt.savefig(file_path)
        plt.show()

    def save(self):
        self.saved = SavedVariables(self.i_h_weigths,self.h_o_weigths , self.h_bias , self.o_bias)
    
    def recover(self):
        self.saved.recover(self)