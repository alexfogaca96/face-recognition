from skimage.feature import hog
from skimage.feature import local_binary_pattern

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

    def apply_lpb(self):
        self.apply_function_to_data(self.lpb_func)

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

    @staticmethod
    def lpb_func(data_item):
        partial_one = local_binary_pattern(data_item.face_one ,3,24)
        partial_two = local_binary_pattern(data_item.face_two,3,24 )
        data_item.face_one = partial_one.reshape( partial_one.size)
        data_item.face_two = partial_two.reshape( partial_two.size)

    def get_all_training_data(self):
        all_training_data = []
        for key in self._training_db.keys():
            all_training_data.extend(self._training_db[key])
        return all_training_data