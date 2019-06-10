import cv2
import os
import matplotlib.image


# Returns the default frontal face classifier from cv2
def face_classifier():
    return cv2.CascadeClassifier('D:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')


# LBP Classifier code (https://www.superdatascience.com/blogs/opencv-face-detection)
def detect_face(f_cascade, colored_img, scale_factor=1.1):
    img_copy = colored_img.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray_img)
    faces = f_cascade.detectMultiScale(gray_equalized, scaleFactor=scale_factor, minNeighbors=5)
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
match_pair = 0
match_path = '..\\scrapper\\matches'
for directory in range(3000):
    match_pair += 1

    match_one_path = match_path + '\\' + str(directory) + '\\match_one.jpg'
    match_two_path = match_path + '\\' + str(directory) + '\\match_two.jpg'

    match_one_img = cv2.imread(match_one_path)
    match_two_img = cv2.imread(match_two_path)
    
    if match_one_img is None or match_two_img is None:
        print(match_one_path)
        print(match_two_path)
        continue

    face_detected_img_one, face_one = detect_face(face_cascade, match_one_img)
    face_detected_img_two, face_two = detect_face(face_cascade, match_two_img)

    if len(face_one) == 0 or len(face_two) == 0:
        continue

    face_cropped_img_one = crop_image_by_face(match_one_img, face_one)
    face_cropped_img_two = crop_image_by_face(match_two_img, face_two)

    match_folder = matches_path + '/' + str(match_pair)
    if not os.path.isdir(match_folder):
        os.mkdir(match_folder)

    matplotlib.image.imsave(match_folder + '/match_one.jpg', face_cropped_img_one)
    matplotlib.image.imsave(match_folder + '/match_two.jpg', face_cropped_img_two)


mismatches_path = 'mismatches'
if not os.path.isdir(mismatches_path):
    os.mkdir(mismatches_path)
mismatch_pair = 0
mismatch_path = '..\\scrapper\\mismatches'
for directory in range(3000):
    mismatch_pair += 1

    mismatch_one_path = mismatch_path + '\\' + str(directory) + '\\mismatch_one.jpg'
    mismatch_two_path = mismatch_path + '\\' + str(directory) + '\\mismatch_two.jpg'

    mismatch_one_img = cv2.imread(mismatch_one_path)
    mismatch_two_img = cv2.imread(mismatch_two_path)

    if mismatch_one_img is None or mismatch_two_img is None:
        print(mismatch_one_path)
        print(mismatch_two_path)
        continue

    face_detected_img_one, face_one = detect_face(face_cascade, mismatch_one_img)
    face_detected_img_two, face_two = detect_face(face_cascade, mismatch_two_img)

    if len(face_one) == 0 or len(face_two) == 0:
        continue

    face_cropped_img_one = crop_image_by_face(mismatch_one_img, face_one)
    face_cropped_img_two = crop_image_by_face(mismatch_two_img, face_two)

    mismatch_folder = mismatches_path + '/' + str(mismatch_pair)
    if not os.path.isdir(mismatch_folder):
        os.mkdir(mismatch_folder)

    matplotlib.image.imsave(mismatch_folder + '/mismatch_one.jpg', face_cropped_img_one)
    matplotlib.image.imsave(mismatch_folder + '/mismatch_two.jpg', face_cropped_img_two)
