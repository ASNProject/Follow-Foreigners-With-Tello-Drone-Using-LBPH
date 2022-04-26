import re
import cv2
import os
import numpy as np
from djitellopy import tello
import time

me = tello.Tello()
me.connect()
me.streamon()
me.takeoff()
me.send_rc_control(0,0,20,0)
time.sleep(1.2)

folder_name = 'faces_data'

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()  
recognizer.read('recognizer/faces_data.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

known_names = ['Arief Setyo Nugroho']
Output = 'Tidak Diketahui'

w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0

drone = 0

def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y , w , h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)

        roi_gray = imgGray[y:y+h, x:x+w]
        ids, dist = recognizer.predict(roi_gray)



        if (dist<50):
            cv2.putText(img, f'{known_names[ids-1]} {round(dist,2)}', (x-20, y-20), font, 1 , (255, 255, 0), 3)
            #print('Known Faces')
            drone = 0
        else:
            cv2.putText(img, f'{Output} {round(dist,2)}', (x-20, y-20), font, 1 , (255, 255, 0), 3)
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            cv2.circle(img, (cx, cy), 5, (0,0,255), cv2.FILLED)
            myFaceListC.append([cx,cy])
            myFaceListArea.append(area)
            #print("Drone Follow Foreifners")

    if len(myFaceListArea) !=0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    
    else:
        return img, [[0,0],0]
        
def trackFace(info, w, pid, pError):
    area = info[1]
    x,y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area !=0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    me.send_rc_control(0, fb, 0, speed)
    return error

#cap = cv2.VideoCapture(0)
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w,h))
    img, info = findFace(img)
    pError = trackFace(info, w, pid, pError)
    print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)
    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break
    elif key == ord('w'):
        me.move_forward(30)
    elif key == ord('s'):
        me.move_back(30)
    elif key == ord('a'):
        me.move_left(30)
    elif key == ord('d'):
        me.move_right(30)
    elif key == ord('e'):
        me.rotate_clockwise(30)
    elif key == ord('q'):
        me.rotate_counter_clockwise(30)
    elif key == ord('r'):
        me.move_up(30)
    elif key == ord('f'):
        me.move_down(30)
    elif key == ord('x'):
        me.land()