import tensorflow as tf
import tkinter as tk
from PIL import Image as PLIMG
from PIL import ImageTk
import cv2 as cv
import os
import numpy as np
import pyautogui
import pandas as pd
from win32api import GetSystemMetrics
from cv2 import CascadeClassifier
from sklearn.model_selection import train_test_split

class Prediction():
    def __init__(self):
        self.trainX, self.testX, self.trainY,self.testY=0,0,0,0
        #self.fit()
        self.dada,self.label=[],[]
    def data(self,data,labels):
        self.trainX, self.testX, self.trainY,self.testY = train_test_split(data, labels, test_size=0.1)
        self.trainX=np.array(self.trainX)/255.0
        self.trainY=np.array(self.trainY)
        self.testX=np.array(self.testX)/255.0
        self.testY=np.array(self.testY)
    def fit(self):
        self.model= tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(kernel_size= 5,filters= 20,strides=(1, 1),activation='relu', input_shape=(32,32, 3)))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))                                             
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(units=2, activation='tanh'))
        self.model.compile(tf.keras.optimizers.Adam(0.0005),loss="mean_squared_error")
        self.H=self.model.fit(self.trainX, self.trainY, validation_data=(self.testX,self.testY),epochs=150, batch_size=32)
        print("Model trained")
    def prediction(self,image):
        Image=Image()
        Image.eye_search(self.image)
        self.eye=Image.frame
        if type(eye)!="<class 'int'>":
            self.prediction=self.model.predict(tf.expand_dims(self.image, axis=0))
            return prediction

class Image():
    def __init__(self):
        self.img_w_h=32
        self.faceCascade=0
        self.faceCascade=0
        self.faces=0
        self.color="#ff0000"
        self.moments=(0,0,0)
        self.mask=0
        self.contours=0
        self.num=0
        self.show_frame=0
    def eye_search(self,frame):
        self.faceCascade = CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
        self.eye_cascade = CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        self.faces=self.faceCascade.detectMultiScale(frame,scaleFactor=1.1,
            minNeighbors=5, minSize=(130, 130), flags=cv.CASCADE_SCALE_IMAGE)
        self.frame=10
        for (x, y, w, h) in self.faces:
            self.eye=self.eye_cascade.detectMultiScale(frame,scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30), maxSize=(60, 60), flags=cv.CASCADE_SCALE_IMAGE)
            if len(self.faces)==1 and len(self.eye)==2:
                self.eye=list(self.eye)
                (x1,y1,w1,h1)=tuple(self.eye[0])
                (x2,y2,w2,h2)=tuple(self.eye[1])                  
                self.frame=frame[y1+6:y1+w1-6, x1+6:x1+h1-6]
                if x1<x2:
                    if x1>x and y1>y and y1+w1<y+w and x1+h1<x+h:
                        self.frame=frame[y1+6:y1+w1-6, x1+6:x1+h1-6]
                else:
                    if x2>x and y2>y and y2+w2<y+w and x2+h2<x+h:
                        self.frame=frame[y2+6:y2+w2-6, x2+6:x2+h2-6]              
                self.mask = cv.inRange(cv.cvtColor(self.frame, cv.COLOR_BGR2HSV), np.array([100,93,0]),np.array([179,228,110]))
                self.contours, self.hierarchy = cv.findContours(self.mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                for i in range(len(self.contours)):
                    if (cv.arcLength(self.contours[i], True) > 20):
                        self.moments = cv.moments(self.contours[i])
                        if self.moments['m00']>0.0:
                            self.color="#3da859"
                            self.num+=1
                            self.frame=cv.resize(self.frame,(32,32))
                            self.show_frame=self.frame
                            self.frame=np.array(self.frame)/255.0
                            return self.frame                              
            else:
                return 10

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.Image=Image()
        self.cap=cap
        self._,self.frame = cap.read()
        self.Image.eye_search(self.frame)
        self.width = GetSystemMetrics(0)
        self.height = GetSystemMetrics(1)
        print(GetSystemMetrics(0),GetSystemMetrics(1))
        self.interval = 20 # Interval in ms to get the latest frame
        self.Prediction=Prediction()
        self.eye=0
        self.num=0
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.x1, self.y1, self.x2, self.y2=0,0,0,0
        self.canvas.grid(row=0, column=0)
        self.labels=[]
        self.data=[]
        if self.num<=200:
            self.learning()
        else:
            self.update_image()
    def update_image(self):
        # Get the latest frame and convert image format
        self._,self.frame = self.cap.read()
        self.Image.eye_search(self.frame)
        self.eye=self.Image.show_frame
        if str(type(self.Image.frame))!="<class 'int'>":
            if self.Image.frame.shape==(32,32,3):
                self.prediction=self.Prediction.model.predict(tf.expand_dims(self.Image.frame, axis=0))
                self.draw_points(self.prediction[0][0],self.prediction[0][1])
                print(self.prediction)
        if str(type(self.Image.show_frame))=="<class 'numpy.ndarray'>":
              self.image = PLIMG.fromarray(self.Image.show_frame)# to PIL format
              self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
              self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.window.after(self.interval, self.update_image)
      
    def draw_points(self,x,y):
        self.x1, self.y1 = (x*1920), (y*1080)
        self.x2, self.y2 = ((x*1920)+7), ((y*1080)+7)
        self.canvas.create_oval(self.x1, self.y1, self.x2, self.y2, fill=self.Image.color) 
    def point(self,event):
        self.num+=1
        self.x1, self.y1 = (event.x), (event.y)
        self.x2, self.y2 = (event.x + 7), (event.y + 7)
        self.labels.append([self.x1/1920,self.y1/1080])
        self.data.append(self.eye)
        self.canvas.create_oval(self.x1, self.y1, self.x2, self.y2, fill=self.Image.color)
    def learning(self):
        self._,self.frame = self.cap.read()
        self.Image.eye_search(self.frame)
        self.eye=self.Image.show_frame
        if str(type(self.Image.show_frame))=="<class 'numpy.ndarray'>":
              self.image = PLIMG.fromarray(self.Image.show_frame)# to PIL format
              self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
              self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
              self.canvas.create_text(100,0,fill="darkblue",font=str(self.num))
        self.canvas.bind("<Button-1>", self.point)
        if self.num==200:
              self.Prediction.data(self.data,self.labels)
              self.Prediction.fit()
              print(self.num)
              self.window.after(self.interval, self.update_image)
        else:
              self.window.after(self.interval, self.learning)
if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root,cv.VideoCapture(0,cv.CAP_DSHOW))
    root.mainloop()
