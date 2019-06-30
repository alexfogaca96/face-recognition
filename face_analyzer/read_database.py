import os
from enumerators.match_type import MatchType
from pathlib import Path
import cv2 
from entities.data_item import DataItem
import time
import random
from entities.training_validation_database import TrainingValidationDatabase
import numpy as np

class ReadDatabase():
    def __init__(self):
        self.all_data = None 

    def read(self):
        folders = [ [MatchType.MATCH , Path('../all_cropped_faces/Match/')] ,
                    [MatchType.MISMATCH , Path('../all_cropped_faces/Mismatch/')] 
        ]
        self.all_data = []
        for type_folder in  folders:
            match_type , folder_path = type_folder[0] , type_folder[1]
            self.all_data.extend(self.read_folder(match_type , folder_path))
        random.shuffle(self.all_data)

    def read_folder(self , match_type , folder_path):
        list_data_item = [] 
        for faces_folder in os.listdir(folder_path):
            face_folder_path = folder_path / faces_folder
            list_face = [] 
            for face in os.listdir(face_folder_path):
                dim = (73 ,73)
                image_path = face_folder_path / face
                face_image = cv2.imread(str(image_path))
                face_image = cv2.resize(face_image, dim, interpolation = cv2.INTER_AREA)
                list_face.append(face_image)
            data_item = DataItem(list_face[0] , list_face[1] , match_type)
            list_data_item.append(data_item)
        return list_data_item
    
    def get_k_fold_database(self , epoch_num , training_percentage):
        training_qtd = int(len(self.all_data) * training_percentage)
        all_data_copy = self.all_data.copy()
        random.shuffle(all_data_copy)
        training_data = all_data_copy[:training_qtd]
        validation_data = all_data_copy[training_qtd:]
        del all_data_copy
        all_training_lists = self.divide_data_in_n(training_data , epoch_num)
        del training_data
        return TrainingValidationDatabase(all_training_lists , validation_data)
        
    
    def divide_data_in_n(self , all_data , num_chunks):
        all_chuncks = {}
        for index , data  in enumerate(all_data):
            new_index = index % num_chunks
            all_chuncks.setdefault(new_index , []).append(data)
        return all_chuncks
            