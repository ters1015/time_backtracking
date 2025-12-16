import face_recognition
import cv2
import numpy as np
from PIL import Image


def face_rec_user(image):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="hog")
    face_encoding = face_recognition.face_encodings(img, face_locations)
    if face_encoding:
        know_encoding = face_encoding[0]
        know_encoding = know_encoding.tolist()
        return know_encoding
    else:
        return []


if __name__ == '__main__':
    img = Image.open('/mnt/DataDrive5/zte/preprocess/extract_test/64_26.jpg')
    print(face_rec_user(img))
