import os 
import sys
from pathlib import Path
path = Path(os.path.realpath(__file__))
parent_folder = path.parents[1] / "Shared"
sys.path.insert(0 , str(parent_folder))
from read_database import ReadDatabase
from knn_runner import KNNRunner

def main():
    k_folders_validation = 5
    read_database = ReadDatabase()
    print("reading database... ")
    read_database.read()
    print("breaking database in " + str(k_folders_validation) + " training databases and 1 testing database")
    training_validation = read_database.get_k_fold_database(k_folders_validation, 0.7)
    print("applying hog to database...")
    training_validation.apply_hog()
    KNNRunner().run([1,2,3,4,5,6,7,8,9,10,15,20,30] , training_validation )

if __name__ == "__main__":
    main()
