import cv2
import os
from pathlib import Path
import matplotlib.image
from entities.face_pair import FacePair
from entities.face import Face
from enumerators.match_type import MatchType
import pdb

def save_all_pairs(list_all_pairs):
    main_path = Path('../all_cropped_faces/')
    if not os.path.exists(main_path):
        os.mkdir(main_path)

    for match_type in MatchType:
        type_path = main_path / match_type.value
        if not os.path.exists(type_path):
            os.mkdir(type_path)
        pairs_list = list(filter(lambda x: x.match_type == match_type ,list_all_pairs ))

        for pair in pairs_list:
            pair_folder_path = type_path / pair.origin_folder_name
            if not os.path.exists(pair_folder_path):
                os.mkdir(pair_folder_path)
            for face in pair.faces:
                matplotlib.image.imsave(pair_folder_path / face.image_name , face.face_image)
                print("generating cropped image in " + str(pair_folder_path / face.image_name))

def extract_faces_from_folder( folder_name , face_classifier, match_type):
    all_pairs = [] 
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    match_path = Path(folder_name)
    for  directory in os.listdir(folder_name):
        failed_to_find_face = False

        pair_path = match_path /  str(directory)
        detected_faces = [] 
        for image_name in os.listdir(pair_path):
            full_image = cv2.imread(str(pair_path / image_name))
            face_detected_img , face = detect_face(face_classifier, full_image)
            processed_img = None
            if len(face) == 0 or failed_to_find_face:
                failed_to_find_face = True
                processed_img = full_image
            else:
                processed_img = crop_image_by_face(full_image, face)
            face_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            detected_faces.append(Face(face_img , image_name))
               
        face_pair = None

        if failed_to_find_face:
            face_pair = FacePair(detected_faces[0] , detected_faces[1], MatchType.FACE_NOT_FOUND,  directory+"_"+match_type.value )
        else:
            face_pair = FacePair(detected_faces[0] , detected_faces[1], match_type ,  directory)
        
        all_pairs.append(face_pair)

    return all_pairs
    
# Returns the default frontal face classifier from cv2
def face_classifier():
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))
    return cv2.CascadeClassifier(str(current_path / "opencv_sources/haarcascade_frontalface_alt2.xml"))


# Viola Jones Classifier code (https://www.superdatascience.com/blogs/opencv-face-detection)
def detect_face(f_cascade, colored_img, scale_factor=1.05):
    img_copy = colored_img.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    #gray_equalized = cv2.equalizeHist(gray_img)
    faces = f_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=1 , minSize =(60,60))
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_copy, faces


# Uses the vertices of the detected face on the image to crop and overwrite it
def crop_image_by_face(original_img, faces):
    face = faces[0]
    return original_img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]


# Show given images and wait for all of them to be closed
def show_images(images):
    count = 0
    for img in images:
        cv2.imshow('image ' + str(count), img)
        count += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":    
    # Initialize classifier
    face_cascade = face_classifier()
    all_pairs = []
    match_path = Path('../scrapper/matches')
    mismatch_path = Path('../scrapper/mismatches')
    all_pairs.extend(extract_faces_from_folder(match_path , face_cascade, MatchType.MATCH))
    all_pairs.extend(extract_faces_from_folder(mismatch_path , face_cascade, MatchType.MISMATCH))
    save_all_pairs(all_pairs)
    
