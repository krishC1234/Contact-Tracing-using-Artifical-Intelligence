#Written by Krish Chaudhary and published on August 22, 2020

import cv2
import os
import numpy as np
from PIL import Image
import pickle
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = "images"

face_cascade = cv2.CascadeClassifier('face.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

for labels in os.listdir(image_dir):
	try:
		for images in os.listdir(image_dir+'/'+labels):
			if not images.startswith('.'):
				print("hello")
				y_labels.append(current_id)
				pil_image = Image.open(image_dir+'/'+labels+'/'+images).convert("L") # grayscale
				x_train.append(np.array(pil_image, "uint8"))
		current_id += 1
	except:
		pass


with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

