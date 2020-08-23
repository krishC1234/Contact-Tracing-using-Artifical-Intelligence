#Written by Krish Chaudhary and published on August 22, 2020

import numpy as np
import cv2
import pickle
from mtcnn.mtcnn import MTCNN
import random


cap = cv2.VideoCapture('edited.mov')
ij=0 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector=MTCNN()
    faces = detector.detect_faces(frame)
    print(faces)

    for face in faces:
        print(face['box'])
        for i in face['box']:
            (x,y,w,h)=face['box']
        #roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        name = str(random.randint(0,2000))
        cv2.imwrite('../keras-retinanet/images/'+name+'.jpg', roi_color)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if(ij>17):
        break
    else:
        ij+=1
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()