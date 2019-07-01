import os 
import sys
from pathlib import Path
path = Path(os.path.realpath(__file__))
parent_folder = path.parents[1] / "Shared"
sys.path.insert(0 , str(parent_folder))
from read_database import ReadDatabase
from neural_network import NeuralNetwork
from activation_functions import ActivationFunctions




if __name__ == "__main__":
    k_folders_validation = 5
    read_database = ReadDatabase()
    print("reading database... ")
    read_database.read()
    print("breaking database in " + str(k_folders_validation) + " training databases and 1 testing database")
    training_validation = read_database.get_k_fold_database(k_folders_validation, 0.7)
    print("applying hog to database...")
    training_validation.apply_hog()
    example_data = training_validation.get_training_database(0)[0]
    neural_network =NeuralNetwork(12 ,ActivationFunctions.sigmoid_function , len(example_data.face_one)*2 , 2 , 0.1)
    neural_network.execute_training(training_validation)
