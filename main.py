import streamlit as st
#import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import operator
import time
import sys
import os
import matplotlib.pyplot as plt
from string import ascii_uppercase
from PIL import Image, ImageTk

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")


json_file_dru = open("model-bw_dru.json" , "r")
model_json_dru = json_file_dru.read()
json_file_dru.close()
loaded_model_dru = model_from_json(model_json_dru)
loaded_model_dru.load_weights("model-bw_dru.h5")
json_file_tkdi = open("model-bw_tkdi.json" , "r")
model_json_tkdi = json_file_tkdi.read()
json_file_tkdi.close()
loaded_model_tkdi = model_from_json(model_json_tkdi)
loaded_model_tkdi.load_weights("model-bw_tkdi.h5")
json_file_smn = open("model-bw_smn.json" , "r")
model_json_smn = json_file_smn.read()
json_file_smn.close()
loaded_model_smn = model_from_json(model_json_smn)
loaded_model_smn.load_weights("model-bw_smn.h5")
    




def predict(test_image):
    test_image = cv2.resize(test_image, (128,128))
    result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))
    result_dru = loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
    result_tkdi = loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
    result_smn = loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
    prediction={}
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]

    
    if(current_symbol == 'D' or current_symbol == 'R' or current_symbol == 'U'):
        prediction = {}
        prediction['D'] = result_dru[0][0]
        prediction['R'] = result_dru[0][1]
        prediction['U'] = result_dru[0][2]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

    if(current_symbol == 'D' or current_symbol == 'I' or current_symbol == 'K' or current_symbol == 'T'):
        prediction = {}
        prediction['D'] = result_tkdi[0][0]
        prediction['I'] = result_tkdi[0][1]
        prediction['K'] = result_tkdi[0][2]
        prediction['T'] = result_tkdi[0][3]
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

    if(current_symbol == 'M' or current_symbol == 'N' or current_symbol == 'S'):
        prediction1 = {}
        prediction1['M'] = result_smn[0][0]
        prediction1['N'] = result_smn[0][1]
        prediction1['S'] = result_smn[0][2]
        prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        if(prediction1[0][0] == 'S'):
            current_symbol = prediction1[0][0]
        else:
            current_symbol = prediction[0][0]
    return current_symbol
   
st.title("Sign Langauge Detection")

run = st.checkbox('Real time')


cap=cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

FRAME_WINDOW = st.image([])

while run:
        ok, frame = cap.read()    
        
        cv2image = frame
        x1 = 50
        y1 = 50
        x2 = 300
        y2 = 300
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,0)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        cv2image = cv2image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        result=predict(res)

        FRAME_WINDOW.image(cv2.putText(frame,result, (50, 50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),1))
        FRAME_WINDOW.image(frame)

        
else:    
            st.subheader("stopped")

cap.release()
cv2.destroyAllWindows()
