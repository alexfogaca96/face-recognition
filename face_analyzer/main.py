from read_database import ReadDatabase
from neural_network import NeuralNetwork
from activation_functions import ActivationFunctions

if __name__ == "__main__":
    kfolder_validation = 5
    read_database = ReadDatabase()
    print("reading database... ")
    read_database.read()
    print("breaking database in " + str(kfolder_validation) + " training databases and 1 testing database")
    training_validation = read_database.get_k_fold_database(kfolder_validation , 0.7)
    print("applying hog to database...")
    training_validation.apply_hog()
    data = training_validation.get_training_database(1)[0]
    neural_network =NeuralNetwork(6 ,ActivationFunctions.sigmoid_function , len(data.face_one)*2)
    neural_network.calculate_first_layer_output(data.face_one, data.face_two)
