import os
import cv2
import random
from pathlib import Path
from face_analyzer.enumerators.match_type import MatchType
from face_analyzer.entities.data_item import DataItem
from face_analyzer.entities.training_validation_database import TrainingValidationDatabase


class ReadDatabase:
    def __init__(self):
        self.all_data = None 

    def read(self):
        folders = [[MatchType.MATCH, Path('../all_cropped_faces/Match/')],
                   [MatchType.MISMATCH, Path('../all_cropped_faces/Mismatch/')]]
        self.all_data = []
        for type_folder in  folders:
            match_type , folder_path = type_folder[0] , type_folder[1]
            self.all_data.extend(self.read_folder(match_type , folder_path))
        random.shuffle(self.all_data)

    @staticmethod
    def read_folder(match_type, folder_path):
        list_data_item = [] 
        for faces_folder in os.listdir(folder_path):
            face_folder_path = folder_path / faces_folder
            list_face = [] 
            for face in os.listdir(face_folder_path):
                dim = (73, 73)
                image_path = face_folder_path / face
                face_image = cv2.imread(str(image_path))
                face_image = cv2.resize(face_image, dim, interpolation=cv2.INTER_AREA)
                list_face.append(face_image)
            data_item = DataItem(list_face[0], list_face[1], match_type)
            list_data_item.append(data_item)
        return list_data_item
    
    def get_k_fold_database(self, epoch_num, training_percentage):
        training_qtd = int(len(self.all_data) * training_percentage)
        all_data_copy = self.all_data.copy()
        random.shuffle(all_data_copy)
        training_data = all_data_copy[:training_qtd]
        validation_data = all_data_copy[training_qtd:]
        del all_data_copy
        all_training_lists = self.divide_data_in_n(training_data, epoch_num)
        del training_data
        return TrainingValidationDatabase(all_training_lists, validation_data)

    @staticmethod
    def divide_data_in_n(all_data, num_chunks):
        all_chunks = {}
        for index, data in enumerate(all_data):
            new_index = index % num_chunks
            all_chunks.setdefault(new_index, []).append(data)
        return all_chunks
