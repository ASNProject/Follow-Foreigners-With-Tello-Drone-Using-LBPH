import cv2 
import os
from matplotlib import image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

images = os.listdir('faces_data')

image_arrays = []
image_ids = []

for image_path in images:
    splitted_path = image_path.split('.')
    print(splitted_path)
    image_id = int(splitted_path[1])

    image = cv2.imread(os.path.join('faces_data', image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_array = np.array(image, 'uint8')

    image_arrays.append(image_array)
    image_ids.append(image_id)

recognizer.train(image_arrays, np.array(image_ids))
recognizer.save('recognizer/faces_data.yml')
print('[INFO] TRAINING SUCCESSFULL!!')