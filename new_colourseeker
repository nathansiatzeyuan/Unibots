import cv2
from picamera2 import Picamera2
import time
import numpy as np
import serial


ser = serial.Serial('/dev//ttyACM0', 9600, timeout=1)


pc = Picamera2()
dispW=640
dispH=360

pc.preview_configuration.main.size=(dispW,dispH)
pc.preview_configuration.main.format="RGB888"
pc.preview_configuration.align()
pc.configure("preview")
pc.start()
i = 0
lowerBound = np.array([25,100,100]) # Colour range for filtering for yellow balls
upperBound = np.array([35,255,255])

lowerBound1 = np.array([0,0,200]) # Colour range for filtering for white balls
upperBound1 = np.array([180,30,255])

while True:
    frame = pc.capture_array()
    frameBlur = cv2.medianBlur(frame, 15)
    #can also apply median or bilateral blurring based on iteration
    frameHSV=cv2.cvtColor(frameBlur,cv2.COLOR_BGR2HSV)
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    #myMask = cv2.erode(myMask, None, iterations = 2)
    #myMask = cv2.dilate(myMask, None, iterations = 2)
    #closest_centre_distance = 10000
    x_distances = []
    cv2.line(frame, (420, 0), (420, dispH), (0, 255, 0), thickness=2)
    cv2.line(frame, (220, 0), (220, dispH), (0, 255, 0), thickness=2)
    
    contours, hierarchy = cv2.findContours(myMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, width, height = cv2.boundingRect(c)
        if (y+(height/2)) > dispH/2 -200 and 250<(height*width)<10000: #ASSUME area=100x100
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
            centre_distance = int(dispW / 2) - (x + width/2)
            x_distances.append([abs(centre_distance), centre_distance])
        else:
            pass
    x_distances.sort()
    
    #send x_distances to arduino
    if len(x_distances) > 0:
        closest_distance = x_distances[0][1]
        ser.write(str(closest_distance).encode())
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        #time.sleep(1)
    
    cv2.imshow("Camera", frame)
    cv2.imshow('my Mask',myMask)
    
    if cv2.waitKey(1)==ord('q'):
        break
        

cv2.destroyAllWindows()
ser.close()
