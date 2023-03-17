import apriltag
import argparse
import cv2
import numpy as np
from pupil_apriltags import Detector

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

params = (314.22174729465604, 311.4202447283487, 337.0278425306902, 238.99954338265644)

print("got image")
detector = Detector(families="tag36h11")
results = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=0.15)
print(format(len(results))+ " apriltags detected")
#print(results)
tags = {3: (np.array([[-0.3],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]])), 0: (np.array([[0],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]])),
        1: (np.array([[0.3],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]])), 2: (np.array([[0],[0],[0]]) ,np.array([[1,0,0],[0,1,0],[0,0,1]]))}  #id : ( vector t, matrix R)
sum = np.array([[0],[0],[0]])
for r in results:
    est_position =  tags[r.tag_id][0] + tags[r.tag_id][1] @ r.pose_R.T @ r.pose_t * -1
#    print("tagid")
 #   print(r.tag_id)
  #  print("position:")
   # print(est_position)
    sum = sum + est_position
print("position:")
print(sum/len(results))
