import cv2
from picamera2 import Picamera2
import time
import numpy as np
import serial


ser = serial.Serial('dev/ttyACM0', 9600, timeout=1)
sleep(2)
pc = Picamera2()
dispW=1280
dispH=720

pc.preview_configuration.main.size=(dispW,dispH)
pc.preview_configuration.main.format="RGB888"
pc.preview_configuration.align()
pc.configure("preview")
pc.start()

lowerBound=np.array([10,100,100])
upperBound=np.array([30,255,255])
while True:
    frame = pc.capture_array()
    frameBlur= cv2.blur(frame, (15,15))
    frameBlur = cv2.GaussianBlur(frameBlur, (11,11), 0)
    #can also apply median or bilateral blurring based on iteration
    frameHSV=cv2.cvtColor(frameBlur,cv2.COLOR_BGR2HSV)
    
    
    
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    #myMask = cv2.erode(myMask, None, iterations = 2)
    #myMask = cv2.dilate(myMask, None, iterations = 2)
    #closest_centre_distance = 10000
    x_distances = []
    contours, hierarchy = cv2.findContours(myMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, width, height = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
        centre_distance = int(dispW / 2) - (x + width/2)
        x_distances.append([abs(centre_distance), centre_distance])
    x_distances.sort()
    if x_distances[0][1] > 5:
        ser.write(b'1') #turn left
    if x_distances[0][1] < -5 :
        ser.write(b'2')  #turn right
    if x_distances[0][1] <= 5 and x_distances[0][1] >= -5:
        ser.write(b'3')  #forward
    if len(x_distances) == 0:
        ser.write(b'4')  #rotate and scan; executed by arduino
    
   # myMaskSmall=cv2.resize(myMask,(int(dispW/2),int(dispH/2)))
   # myObject=cv2.bitwise_and(frame, frame, mask=myMask)
 #   myObjectSmall=cv2.resize(myObject,(int(dispW/2),int(dispH/2)))
    cv2.imshow("Camera", frame)
    cv2.imshow('my Mask',myMask)
 #   cv2.imshow('My Objest',myObjectSmall)
    if cv2.waitKey(1)==ord('q'):
        break
    