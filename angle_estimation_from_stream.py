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
tags = {3: (np.array([[-0.3],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]]), 0), 0: (np.array([[0],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]]), 0),
        1: (np.array([[0.3],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]]) , 0), 2: (np.array([[0],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]]), 0 )}  #id : ( vector t, matrix R)
params = (314.22174729465604, 311.4202447283487, 337.0278425306902, 238.99954338265644)


while True:
    frame = pc.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=0.15)
    theta_x = 0
    theta_z = 0
   
    theta_final = 0
    for r in results:
        x_rot_vector =  [r.pose_R[0][0], r.pose_R[2][0]]
        est_angle_x = np.arccos(1/(np.sqrt(x_rot_vector[0]**2+ x_rot_vector[1]**2)) * (x_rot_vector[0]))
        theta_x = theta_x + est_angle_x/len(results)
        
        z_rot_vector =  [r.pose_R[0][2], r.pose_R[2][2]]
        est_angle_z = np.arccos(1/(np.sqrt(z_rot_vector[0]**2+ z_rot_vector[1]**2)) * (z_rot_vector[1]))
        theta_z = theta_z + est_angle_z/len(results)
  
        theta_final = tags[r.tag_id][2] + ((theta_z + theta_x)/2)
    
        
      
    #cv2.putText(frame, "theta_x:"+ str(theta_x), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, "theta_final:"+ str(theta_final), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    cv2.imshow("camera feed", frame)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()





