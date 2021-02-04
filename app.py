import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from models.Fac_Model import Fac_Model
from data_preprocess import load_data, webcam_img_process
import numpy as np

model = keras.models.load_model('./Fac_Model')
cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('./util/haarcascade_frontalface_alt2.xml')

labels_map = { 0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral' }

while 1:

    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.2, 4)

    for x, y, w, h in face:
        cv.rectangle(img, (x,y),(x+w,y+h),(255, 239, 0), 2)
        gray_face = gray[y:y+h, x:x+w]
        proc_img = webcam_img_process(gray_face)

        predict = model.predict(proc_img)
        label = labels_map[np.argmax(predict[0])]
        cv.putText(img, label, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 239, 0), 3, cv.LINE_AA)
        
    cv.imshow('CAM', img)
    if cv.waitKey(1) & 0xff == 27: break

cv.destroyAllWindows()
