#set up the cam and stuff
#will eventually need to set up the lo res stream for concurrent apriltag detection
#while true, frame = capture...
#do all the colour detection stuff
#mask bitwise xor grayscale frame
#circle detection on the xored frame
#process results - draw rectangles, orientate, etc

import cv2
from picamera2 import Picamera2
import numpy as np
pc = Picamera2()
dispW=480
dispH=360

pc.preview_configuration.main.size=(dispW,dispH)
pc.preview_configuration.main.format="RGB888"
pc.preview_configuration.align()
pc.configure("preview")
pc.start()
lowerBound=np.array([11,100,100])#colour range for filtering for non-white balls
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
    contours, hierarchy = cv2.findContours(myMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   
    flippedMask = cv2.bitwise_not(myMask)
    #frame_minus_colour = cv2.bitwise_and(frame, frame, mask=flippedMask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #either frame or frame minus colour
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.6, 100)
    
    if circles is not None:
        circles = np.round(circles[0,:].astype("int"))
        for (x, y, r) in circles:
            
            cv2.circle(frame, (x, y), r, (0,255,0), 3)
    
    for c in contours:
        x, y, width, height = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
    cv2.imshow("Camera", frame)
   # cv2.imshow("cam minus mask", frame_minus_colour)
    cv2.imshow('my Mask',myMask)
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()
    