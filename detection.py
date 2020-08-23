#Written by Krish Chaudhary and published on August 22, 2020

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase

# import miscellaneous modules
import cv2
import os
import numpy as np
import time
import math

from mtcnn.mtcnn import MTCNN

import pymongo
from pymongo import MongoClient
from datetime import datetime

import smtplib 

gpu = 0
setup_gpu(gpu)

detector = MTCNN()



font = cv2.FONT_HERSHEY_SIMPLEX

label_name = ['Hansen_Bush','Jack_Daniel','Ronald_Pickett']


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


model = models.load_model('_pretrained_model.h5', backbone_name='resnet50')

uri='mongodb+srv://krish03:Boombeach2!@cluster0.fzjus.mongodb.net/test?retryWrites=true&w=majority'
client = pymongo.MongoClient(uri)

db2=client.test_2

col = db2.person
col2=db2.numbers

col.delete_many({})


img_list = []
ij = 0
count=0

import math
def get_depth(lineAB,lineAD,lineBC):
  lineAC=lineAB
  lineCD=lineAD-lineAC
  angleBCD=math.degrees(math.acos((lineBC**2 + lineAC**2 - lineAB**2)/(-2.0 * lineBC * lineAC)))
  lineBD=math.sqrt(lineBC**2+lineCD**2-(math.cos(math.radians(angleBCD))*2*lineBC*lineCD))
  # print("The straight line distance between the two objects is", round(lineBD,2),"cm")
  return lineBD

cap = cv2.VideoCapture('stock.mp4')
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( "Number of frames: ",count )
while(cap.isOpened()):
  
  try:
    ret, frame = cap.read()
    ij += 1

    if ij == 3:
        break
  
    # copy to draw on
    draw = frame.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(frame)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start, "\t", ij)

    # correct for image scale
    boxes /= scale

    box_list = []
    mid_list = []
    height_list = []

    desc = 'OK'

    

    # visualize detections
    breached=[]
    bounding_boxes=[]
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        
        if label == 0:
          color = [0 , 255  , 0]
          box_list.append(box.tolist())
          mid_list.append( abs(box[0] + box[2])/2 )
          height_list.append(abs(box[3] - box[1]))

          body_img = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
          faces = detector.detect_faces(body_img)
          for face in faces:
            x = face['box'][0]
            y = face['box'][1]
            w = face['box'][2]
            h = face['box'][3]
            roi_gray = body_img[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
            id_, conf = recognizer.predict(roi_gray)
            if conf>=0 and conf<=50:
              cv2.putText(draw, label_name[int(id_)],(box[0],box[1]), font, 2, (0,0,0),4)
            # cv2.putText(draw, labels[id_]+"\t"+str(conf),(1250,188), font, 1, (0,255,0))
            # cv2.rectangle(draw, (x, y), (x+w, y+h), (255, 0, 0), 2)
          
          b = box.astype(int)
          # draw_box(draw, b, color=color)
          cv2.rectangle(draw,(box[0],box[1]),(box[2],box[3]),(0,255,0),3)
          
          
          font = cv2.FONT_HERSHEY_SIMPLEX
          # cv2.putText(draw, desc ,(box[0],box[1]), font, 1,(255,255,255),2,cv2.LINE_AA)

        for i in range(0, len(mid_list)-1):
          for j in range(i+1, len(mid_list)):
            max_box_height = max( abs(box_list[i][3] - box_list[i][1]), abs(box_list[j][3] - box_list[j][1]) )
            diff = math.sqrt( ( mid_list[i] - mid_list[j] )**2 + ( (box_list[i][1]+box_list[i][3])/2 - (box_list[j][1]+box_list[j][3])/2 )**2 )
            diff = diff*161/max_box_height
            distance1 = 2200*161/( abs(box_list[i][3] - box_list[i][1]) )
            distance2 = 2200*161/(abs(box_list[j][3] - box_list[j][1]))
            
            if get_depth(distance1, distance2 ,diff) < 70:
              human1=''
              human2=''
            # if math.sqrt( ( mid_list[i] - mid_list[j] )**2 + ( (box_list[i][1]+box_list[i][3])/2 - (box_list[j][1]+box_list[j][3])/2 )**2 ) < 1000:
              cv2.line(draw, (int(mid_list[i]), int((box_list[i][1]+box_list[i][3])/2)), (int(mid_list[j]), int((box_list[j][1]+box_list[j][3])/2)), (255, 0, 0) , 3)
              cv2.rectangle(draw,(int(box_list[i][0]),int(box_list[i][1])), (int(box_list[i][2]),int(box_list[i][3])),(255,0,0),3)
              cv2.rectangle(draw,(int(box_list[j][0]),int(box_list[j][1])), (int(box_list[j][2]),int(box_list[j][3])),(255,0,0),3)

              body_img = frame[int(box_list[i][1]):int(box_list[i][3]),int(box_list[i][0]):int(box_list[i][2])]
              faces = detector.detect_faces(body_img)
              for face in faces:
                x = face['box'][0]
                y = face['box'][1]
                w = face['box'][2]
                h = face['box'][3]
                roi_gray = body_img[y:y+h, x:x+w]
                roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
                id_, conf = recognizer.predict(roi_gray)
                #print(label_name[int(id_)])
                #print(breached)
                if (label_name[int(id_)]==None):
                  label_name[int(id_)]='unknown'
                if label_name[int(id_)] not in breached or label_name[int(id_)]=='unknown':
                  breached.append(label_name[int(id_)])
                  human1=label_name[int(id_)]
                  
              
              body_img_2 = frame[int(box_list[j][1]):int(box_list[j][3]),int(box_list[j][0]):int(box_list[j][2])]
              faces = detector.detect_faces(body_img_2)
              for face in faces:
                x = face['box'][0]
                y = face['box'][1]
                w = face['box'][2]
                h = face['box'][3]
                roi_gray = body_img[y:y+h, x:x+w]
                roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
                id_, conf = recognizer.predict(roi_gray)
                #print(label_name[int(id_)])
                #print(breached)
                if (label_name[int(id_)]==None):
                  label_name[int(id_)]='unknown'
                if label_name[int(id_)] not in breached or label_name[int(id_)]=='unknown':
                  breached.append(label_name[int(id_)])
                  human2=label_name[int(id_)]
              
              depth=get_depth(distance1, distance2 ,diff)
              if(human1!=None and human2!=None and human1!="" and human2!=""):
                col.insert_one(
                  {
                    "name":[human1,human2] ,
                    "date": datetime.now() ,
                    "distance": get_depth(distance1, distance2 ,diff),
                    "bounding_box": [box_list[i],box_list[j]] ,
                  }
                )
                timestampStr = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
                print(timestampStr)

                li = [col2.find_one( { "name":human1 } )['email'], col2.find_one( { "name":human2 } )['email']] 
                
                for dest in li: 
                  msg = MIMEMultipart()
                  msg['From'] = "quickprep123@gmail.com"
                  msg['To'] = dest
                  msg['Subject'] = "SOCIAL DISTANCE BREACH"

                  body = "You are in violation of the social distancing rule by "+str(round(depth/30.48,2))+" feet. Specifically, "+human1+" has broken social distancing with "+human2+" on the datetime: "+timestampStr+". This is a warning. Please properly social distance."
                  msg.attach(MIMEText(body,'plain'))

                  text = msg.as_string()
                  server = smtplib.SMTP('smtp.gmail.com',587)
                  server.starttls()
                  server.login("quickprep123@gmail.com", "Boombeach2!")
                  server.sendmail("quickprep123@gmail.com",dest,text)
                  server.quit()


            elif math.sqrt( ( mid_list[i] - mid_list[j] )**2 + ( (box_list[i][1]+box_list[i][3])/2 - (box_list[j][1]+box_list[j][3])/2 )**2 ) < 10000:
              #cv2.line(draw, (int(mid_list[i]), int((box_list[i][1]+box_list[i][3])/2)), (int(mid_list[j]), int((box_list[j][1]+box_list[j][3])/2)), (0, 255, 0) , 2)
              pass
             
    print("List of breached:",breached)
    img_list.append(draw)

  except Exception as e:
    print(e)
    pass

height, width, layers = img_list[0].shape
size = (int(width),int(height))
out = cv2.VideoWriter('final.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for image in img_list:
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  out.write(image)
out.release()
