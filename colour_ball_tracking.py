import cv2
from picamera2 import Picamera2
import time
import numpy as np
pc = Picamera2()
dispW=720
dispH=480

pc.preview_configuration.main.size=(dispW,dispH)
pc.preview_configuration.main.format="RGB888"
pc.preview_configuration.align()
pc.configure("preview")
pc.start()

lowerBound=np.array([21,153,100])
upperBound=np.array([83,255,255])
while True:
    frame = pc.capture_array()
    frameBlur= cv2.blur(frame, (15,15))
    frameBlur = cv2.GaussianBlur(frameBlur, (11,11), 0)
    frameBlur = cv2.medianBlur(frameBlur, 5)
    #can also apply median or bilateral blurring based on iteration
    frameHSV=cv2.cvtColor(frameBlur,cv2.COLOR_BGR2HSV)
    
    
    
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    myMask = cv2.erode(myMask, None, iterations = 2)
    myMask = cv2.dilate(myMask, None, iterations = 2)
    contours, hierarchy = cv2.findContours(myMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, width, height = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
    
    
   # myMaskSmall=cv2.resize(myMask,(int(dispW/2),int(dispH/2)))
   # myObject=cv2.bitwise_and(frame, frame, mask=myMask)
 #   myObjectSmall=cv2.resize(myObject,(int(dispW/2),int(dispH/2)))
    cv2.imshow("Camera", frame)
    cv2.imshow('my Mask',myMask)
 #   cv2.imshow('My Objest',myObjectSmall)
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()