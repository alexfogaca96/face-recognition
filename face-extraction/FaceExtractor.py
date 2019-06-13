import cv2
import os
import matplotlib.image


# Returns the default frontal face classifier from cv2
def face_classifier():
    return cv2.CascadeClassifier('D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml')


# Viola Jones Classifier code (https://www.superdatascience.com/blogs/opencv-face-detection)
def detect_face(f_cascade, colored_img, scale_factor=1.05):
    img_copy = colored_img.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray_img)
    faces = f_cascade.detectMultiScale(gray_equalized, scaleFactor=scale_factor, minNeighbors=4)
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


# Initialize classifier
face_cascade = face_classifier()

matches_path = 'matches'
if not os.path.isdir(matches_path):
    os.mkdir(matches_path)

match_unknown_faces_number = 0
match_unknown_faces = []
match_pair = 0
match_path = '..\\scrapper\\matches'
for directory in range(3000):
    match_pair += 1

    match_pair_path = match_path + '\\' + str(directory)
    pair_files = os.listdir(match_pair_path)
    match_one_img = cv2.imread(match_pair_path + '\\' + pair_files[0])
    match_two_img = cv2.imread(match_pair_path + '\\' + pair_files[1])

    face_detected_img_one, face_one = detect_face(face_cascade, match_one_img)
    face_detected_img_two, face_two = detect_face(face_cascade, match_two_img)

    if len(face_one) == 0 or len(face_two) == 0:
        match_unknown_faces_number += 1
        match_unknown_faces.append(match_pair)
        continue

    face_cropped_img_one = crop_image_by_face(match_one_img, face_one)
    face_cropped_img_two = crop_image_by_face(match_two_img, face_two)

    match_folder = matches_path + '/' + str(match_pair)
    if not os.path.isdir(match_folder):
        os.mkdir(match_folder)

    rgb_img_one = cv2.cvtColor(face_cropped_img_one, cv2.COLOR_BGR2RGB)
    rgb_img_two = cv2.cvtColor(face_cropped_img_two, cv2.COLOR_BGR2RGB)

    matplotlib.image.imsave(match_folder + '/' + pair_files[0], rgb_img_one)
    matplotlib.image.imsave(match_folder + '/' + pair_files[1], rgb_img_two)

    print("saved matched " + str(match_pair))


mismatches_path = 'mismatches'
if not os.path.isdir(mismatches_path):
    os.mkdir(mismatches_path)


mismatch_unknown_faces_number = 0
mismatch_unknown_faces = []
mismatch_pair = 0
mismatch_path = '..\\scrapper\\mismatches'
for directory in range(3000):
    mismatch_pair += 1

    mismatch_pair_path = mismatch_path + '\\' + str(directory)
    pair_files = os.listdir(mismatch_pair_path)
    mismatch_one_img = cv2.imread(mismatch_pair_path + '\\' + pair_files[0])
    mismatch_two_img = cv2.imread(mismatch_pair_path + '\\' + pair_files[1])

    face_detected_img_one, face_one = detect_face(face_cascade, mismatch_one_img)
    face_detected_img_two, face_two = detect_face(face_cascade, mismatch_two_img)

    if len(face_one) == 0 or len(face_two) == 0:
        mismatch_unknown_faces_number += 1
        mismatch_unknown_faces.append(mismatch_pair)
        continue

    face_cropped_img_one = crop_image_by_face(mismatch_one_img, face_one)
    face_cropped_img_two = crop_image_by_face(mismatch_two_img, face_two)

    mismatch_folder = mismatches_path + '/' + str(mismatch_pair)
    if not os.path.isdir(mismatch_folder):
        os.mkdir(mismatch_folder)

    rgb_img_one = cv2.cvtColor(face_cropped_img_one, cv2.COLOR_BGR2RGB)
    rgb_img_two = cv2.cvtColor(face_cropped_img_two, cv2.COLOR_BGR2RGB)

    matplotlib.image.imsave(mismatch_folder + '/' + pair_files[0], rgb_img_one)
    matplotlib.image.imsave(mismatch_folder + '/' + pair_files[1], rgb_img_two)

    print("saved mismatched " + str(mismatch_pair))


print("Match faces unrecognized: " + str(match_unknown_faces_number))
print(match_unknown_faces)

print("Mismatch faces unrecognized: " + str(mismatch_unknown_faces_number))
print(mismatch_unknown_faces)

with open('faces_not_recognized.txt', 'a') as file:
    file.write('match: ')
    for item in match_unknown_faces:
        file.write("%d " % item)
    file.write('\nmismatch: ')
    for item in mismatch_unknown_faces:
        file.write("%d " % item)
