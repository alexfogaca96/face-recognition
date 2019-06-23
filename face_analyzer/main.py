from read_database import ReadDatabase

if __name__ == "__main__":
    kfolder_validation = 5
    read_database = ReadDatabase()
    print("reading database... ")
    read_database.read()
    print("breaking database in " + str(kfolder_validation) + " training databases and 1 testing database")
    training_validation = read_database.get_k_fold_database(kfolder_validation , 0.7)
    print("applying hog to database...")
    training_validation.apply_hog()
    