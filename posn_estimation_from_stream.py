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
tags = {3: (np.array([[-0.3],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]])), 0: (np.array([[0],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]])),
        1: (np.array([[0.3],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]])), 2: (np.array([[0],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]]))}  #id : ( vector t, matrix R)
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





