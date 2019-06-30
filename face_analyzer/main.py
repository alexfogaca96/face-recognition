from face_analyzer.read_database import ReadDatabase
from face_analyzer.neural_network import NeuralNetwork
from face_analyzer.activation_functions import ActivationFunctions
from face_analyzer.k_nearest_neighbors import KNN


def execute_knn(training_data):
    data = []
    for i in range(1, 5):
        data.extend(training_data.get_training_database(i))

    knn = KNN(3, data)
    validation_data = training_data.validation_db
    got_right = 0
    for index, pair in enumerate(validation_data):
        print("testing if pair " + str(index) + " match")
        match = knn.do_they_match(pair.face_one, pair.face_two)
        if match == pair.match_type:
            got_right += 1
    print("Got right " + str(got_right) + " from " + str(len(validation_data)))


def execute_neural_network():
    data = training_validation.get_training_database(1)[0]
    neural_network = NeuralNetwork(6, ActivationFunctions.sigmoid_function, len(data.face_one) * 2)
    neural_network.calculate_first_layer_output(data.face_one, data.face_two)


if __name__ == "__main__":
    k_folders_validation = 5
    read_database = ReadDatabase()
    print("reading database... ")
    read_database.read()
    print("breaking database in " + str(k_folders_validation) + " training databases and 1 testing database")
    training_validation = read_database.get_k_fold_database(k_folders_validation, 0.7)
    print("applying hog to database...")
    training_validation.apply_hog()
<<<<<<< HEAD
    data = training_validation.get_training_database(1)[0]
    neural_network =NeuralNetwork(6 ,ActivationFunctions.sigmoid_function , len(data.face_one)*2 , 2 , 0.7)
    neural_network.execute_training(training_validation)
=======
    print("knn")
    execute_knn(training_validation)
>>>>>>> 8891c50cae05c9b9db49d14e6d92de7992a5a221
