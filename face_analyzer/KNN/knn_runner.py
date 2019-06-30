from k_nearest_neighbors import KNN
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from enumerators.match_type import MatchType
import datetime

class KNNRunner():
    results_path = Path("results")
    def run(self , k_neighbors , training_data):
        data = []
        for i in range(1, 5):
            data.extend(training_data.get_training_database(i))
        if isinstance(k_neighbors , int ):
            self.run_for_k_neighbors(k_neighbors , training_data)
        #checking to see if the value is a iterable
        elif hasattr(k_neighbors, '__iter__'): 
            self.run_for_neighbors_list(k_neighbors , training_data)


    def run_for_k_neighbors(self, k_neighbors , data ):
        training_data = data.get_all_training_data()
        validation_data = data.validation_db
        knn = KNN(k_neighbors, training_data)
        got_right = 0
        for index, pair in enumerate(validation_data):
            print("testing if pair " + str(index) + " match")
            match = knn.do_they_match(pair.face_one, pair.face_two)
            if match == pair.match_type:
                got_right += 1
        print("Got right " + str(got_right) + " from " + str(len(validation_data)))

    def run_for_neighbors_list(self , k_neighbors , data):
        training_data = data.get_all_training_data()
        validation_data = data.validation_db
        knn = KNN(k_neighbors, training_data)
        counter_neighbors= np.array([k_neighbors])
        zeros = np.zeros((counter_neighbors.size,1))
        counter_neighbors = np.transpose(counter_neighbors)
        counter_neighbors = np.append(counter_neighbors , zeros, axis= 1)
        for index, pair in enumerate(validation_data):
            print("testing if pair " + str(index) + " match")
            pair_match_type = pair.match_type
            match = knn.do_they_match(pair.face_one, pair.face_two)
            #removing the column with the number of neighbors
            match = match[:,1]
            print("match_result: " + str(match))
            print("pair_match_type" + str(pair_match_type))
            match = np.array([1 if x == pair_match_type else 0 for x in match])
            counter_neighbors[:,1] = counter_neighbors[:,1] + match
        
        num_neighbors =counter_neighbors[:,0]
        percentage_match =counter_neighbors[:,1]/len(validation_data)
        plt.plot( num_neighbors ,percentage_match )
        plt.ylabel( "Percentage of correct matchs" )
        plt.xlabel( "Number of neighbors")
        file_path = KNNRunner.results_path / ("knn_results" +str(datetime.datetime.now()) + ".pdf")
        plt.savefig(str(file_path))
        plt.show()