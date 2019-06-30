from skimage.feature import hog


class TrainingValidationDatabase:
    def __init__(self, training_db, validation_db):
        self._training_db = training_db
        self.validation_db = validation_db
        self.training_db_size = len(training_db)

    def get_training_database(self, epoch):
        if epoch > self.training_db_size - 1 or epoch < 0:
            raise Exception("Out of Range. Training database has only: " + str(self._training_db.keys()) + " epochs")
        return self._training_db[epoch]

    def apply_hog(self):
        self.apply_function_to_data(self.hog_func)

    def apply_function_to_data(self, func):
        for epoch in self._training_db.keys():
            for data_item in self._training_db[epoch]:
                func(data_item)  
        for data_item in self.validation_db:
            func(data_item)

    @staticmethod
    def hog_func(data_item):
        data_item.face_one = hog(data_item.face_one)
        data_item.face_two = hog(data_item.face_two)

    def get_all_training_data(self):
        all_training_data = []
        for key in self._training_db.keys():
            all_training_data.extend(self._training_db[key])
        return all_training_data