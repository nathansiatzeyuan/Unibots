import apriltag
import argparse
import cv2
import numpy as np
from pupil_apriltags import Detector
from picamera2 import Picamera2, Preview
pc = Picamera2()

pc.preview_configuration.main.size=(1280, 720)
pc.preview_configuration.align()
pc.configure("preview")
pc.start()
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
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()





