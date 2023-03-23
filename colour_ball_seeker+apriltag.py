import cv2 
from picamera2 import Picamera2, Preview
import time
import numpy as np
import serial
import threading
import apriltag
import argparse
from pupil_apriltags import Detector


detector = Detector(families="tag36h11")
North= [[1,0,0],[0,1,0],[0,0,1]]
East= [[0,0,1],[0,1,0],[-1,0,0]]
South= [[-1,0,0],[0,1,0],[0,0,-1]]
West= [[0,0,-1],[0,1,0],[1,0,0]]
tags = {0: (np.array([[150],[0],[0]]) ,np.array(North)), 1: (np.array([[450],[0],[0]]) ,np.array(North)), 2: (np.array([[750],[0],[0]]) ,np.array(North)), 
3: (np.array([[1250],[0],[0]]) ,np.array(North)), 4: (np.array([[1550],[0],[0]]) ,np.array(North)), 5: (np.array([[1850],[0],[0]]) ,np.array(North)),
6: (np.array([[2000],[0],[-150]]) ,np.array(East)), 7: (np.array([[2000],[0],[-450]]) ,np.array(East)), 8: (np.array([[2000],[0],[-750]]) ,np.array(East)),  
9: (np.array([[2000],[0],[-1250]]) ,np.array(East)), 10: (np.array([[2000],[0],[-1550]]) ,np.array(East)), 11: (np.array([[2000],[0],[-1850]]) ,np.array(East)),
12: (np.array([[1850],[0],[-2000]]) ,np.array(South)), 13: (np.array([[1550],[0],[-2000]]) ,np.array(South)), 14: (np.array([[1250],[0],[-2000]]) ,np.array(South)),  
15: (np.array([[750],[0],[-2000]]) ,np.array(South)), 16: (np.array([[450],[0],[-2000]]) ,np.array(South)), 17: (np.array([[150],[0],[-2000]]) ,np.array(South)),
18: (np.array([[0],[0],[-1850]]) ,np.array(West)), 19: (np.array([[0],[0],[-1550]]) ,np.array(West)), 20: (np.array([[0],[0],[-1250]]) ,np.array(West)), 
21: (np.array([[0],[0],[-750]]) ,np.array(West)), 22: (np.array([[0],[0],[-450]]) ,np.array(West)), 23: (np.array([[0],[0],[-150]]) ,np.array(West))}    #id : ( vector t, matrix R)
params = (314.22174729465604, 311.4202447283487, 337.0278425306902, 238.99954338265644)


while True:
    frame = pc.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=0.15)
    position = np.array([[0],[0],[0]])
    for r in results:
        est_position =  tags[r.tag_id][0] + tags[r.tag_id][1] @ r.pose_R.T @ r.pose_t * -1
        position = position + est_position/len(results)
        
    cv2.putText(frame, "coords:"+ str(round(float(position[0]), 2)) +", " + str(round(float(position[1]), 2)) +", " + str(round(float(position[2]),2)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("camera feed", frame)



def send_data(ser, closest_distance):
    ser.write(str(closest_distance).encode())
    line = ser.readline().decode('utf-8').rstrip()
    print(line)


def process_image(pc, ser):
    lower_bound = np.array([25,100,100]) # Colour range for filtering for non-white balls
    upper_bound = np.array([35,255,255])
    dispW = 640
    dispH = 360
    pc.preview_configuration.main.size = (dispW, dispH)
    pc.preview_configuration.main.format = "RGB888"
    pc.preview_configuration.align()
    pc.configure("preview")
    pc.start()
    x_distances = []
    while True:
        
        frame = pc.capture_array()
        frame_blur = cv2.medianBlur(frame, 15)
        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        my_mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
        contours, hierarchy = cv2.findContours(my_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours1=[]
        cv2.line(frame, (420, 0), (420, dispH), (0, 255, 0), thickness=2)
        cv2.line(frame, (220, 0), (220, dispH), (0, 255, 0), thickness=2)
        x_distances = []
        for c in contours:
            x, y, width, height = cv2.boundingRect(c)
            if (y+(height/2)) > dispH/2 -200 and 250<(height*width)<20000: #ASSUME area=100x100
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
                centre_distance = int(dispW / 2) - (x + width/2)
                x_distances.append([abs(centre_distance), centre_distance])
            else:
                pass
        x_distances.sort()
        if len(x_distances) > 0:
            closest_distance = x_distances[0][1]
            threading.Thread(target=send_data, args=(ser, closest_distance)).start()
        cv2.imshow("Camera", frame)
        cv2.imshow('my Mask',my_mask)
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows()
    ser.close()


if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    pc = Picamera2()
    threading.Thread(target=process_image, args=(pc, ser)).start()
