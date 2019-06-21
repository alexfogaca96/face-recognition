class FacePair():
    def __init__(self, first_face , second_face , match_type , origin_folder_name):
        self.faces = [first_face , second_face] 
        self.match_type = match_type
        self.origin_folder_name = origin_folder_name