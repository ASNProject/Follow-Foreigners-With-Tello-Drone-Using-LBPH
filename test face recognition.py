import cv2
import os
import numpy as np

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()  
recognizer.read('recognizer/faces_data.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

known_names = ['Arief Setyo Nugroho']
Output = 'Tidak Diketahui'

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 3)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)

        roi_gray = gray[y:y+h, x:x+w]
        ids, dist = recognizer.predict(roi_gray)
        if (dist<50):
            cv2.putText(frame, f'{known_names[ids-1]} {round(dist,2)}', (x-20, y-20), font, 1 , (255, 255, 0), 3)
        else:
            cv2.putText(frame, f'{Output} {round(dist,2)}', (x-20, y-20), font, 1 , (255, 255, 0), 3)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
