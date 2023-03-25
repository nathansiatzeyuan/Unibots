import cv2 
from picamera2 import Picamera2, Preview
import time
import numpy as np
import serial
import threading
import apriltag
import argparse
from pupil_apriltags import Detector


countdown=20
detector = Detector(families="tag36h11")
North= [[1,0,0],[0,1,0],[0,0,1]]
East= [[0,0,1],[0,1,0],[-1,0,0]]
South= [[-1,0,0],[0,1,0],[0,0,-1]]
West= [[0,0,-1],[0,1,0],[1,0,0]]
North_angle= 0
East_angle= 90
South_angle= 180
West_angle= 270
tags = {0: (np.array([[150],[0],[0]]) ,np.array(North),North_angle),
        1: (np.array([[450],[0],[0]]) ,np.array(North),North_angle),
        2: (np.array([[750],[0],[0]]) ,np.array(North),North_angle), 
        3: (np.array([[1250],[0],[0]]) ,np.array(North),North_angle),
        4: (np.array([[1550],[0],[0]]) ,np.array(North),North_angle),
        5: (np.array([[1850],[0],[0]]) ,np.array(North),North_angle),
        6: (np.array([[2000],[0],[-150]]) ,np.array(East),East_angle),
        7: (np.array([[2000],[0],[-450]]) ,np.array(East),East_angle),
        8: (np.array([[2000],[0],[-750]]) ,np.array(East),East_angle),  
        9: (np.array([[2000],[0],[-1250]]) ,np.array(East),East_angle),
        10: (np.array([[2000],[0],[-1550]]) ,np.array(East),East_angle),
        11: (np.array([[2000],[0],[-1850]]) ,np.array(East),East_angle),
        12: (np.array([[1850],[0],[-2000]]) ,np.array(South),South_angle),
        13: (np.array([[1550],[0],[-2000]]) ,np.array(South),South_angle),
        14: (np.array([[1250],[0],[-2000]]) ,np.array(South),South_angle),  
        15: (np.array([[750],[0],[-2000]]) ,np.array(South),South_angle),
        16: (np.array([[450],[0],[-2000]]) ,np.array(South),South_angle),
        17: (np.array([[150],[0],[-2000]]) ,np.array(South),South_angle),
        18: (np.array([[0],[0],[-1850]]) ,np.array(West),West_angle),
        19: (np.array([[0],[0],[-1550]]) ,np.array(West),West_angle),
        20: (np.array([[0],[0],[-1250]]) ,np.array(West),West_angle), 
        21: (np.array([[0],[0],[-750]]) ,np.array(West),West_angle),
        22: (np.array([[0],[0],[-450]]) ,np.array(West),West_angle),
        23: (np.array([[0],[0],[-150]]) ,np.array(West),West_angle)
        }    #id : ( vector t, matrix R)
params = (314.22174729465604, 311.4202447283487, 337.0278425306902, 238.99954338265644)



def send_data(ser, closest_distance):
    ser.write(str(closest_distance).encode())
    line = ser.readline().decode('utf-8').rstrip()
    print(line)

def capture_frame(pc, delay):
    time.sleep(delay)  # Add a delay between frames
    return pc.capture_array()

def process_image(pc, ser,countdown):
    lower_bound = np.array([25,52,72]) # Colour range for filtering for non-white balls
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
        print(f"Time remaining: {countdown}")
        countdown -= 1
        time.sleep(1)
        frame = capture_frame(pc,delay=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=0.1)
        position = np.array([[0],[0],[0]])
        theta_final = 0
        for r in results:
            est_position =  tags[r.tag_id][0] + tags[r.tag_id][1] @ r.pose_R.T @ r.pose_t * -1
            position = position + est_position/len(results)
            x_rot_vector =  [r.pose_R[0][0], r.pose_R[2][0]]
            est_angle_x = np.arccos(1/(np.sqrt(x_rot_vector[0]**2+ x_rot_vector[1]**2)) * (x_rot_vector[0]))
            #theta_x = theta_x + est_angle_x/len(results)
            
            z_rot_vector =  [r.pose_R[0][2], r.pose_R[2][2]]
            est_angle_z = np.arccos(1/(np.sqrt(z_rot_vector[0]**2+ z_rot_vector[1]**2)) * (z_rot_vector[1]))
            #theta_z = theta_z + est_angle_z/len(results)
      
            #theta_final = np.rad2deg(tags[r.tag_id][2] + ((theta_z + theta_x)/2))
            theta_final = theta_final + ((est_angle_x + est_angle_z)/2 + tags[r.tag_id][2])/len(results)
        cv2.putText(frame, "coords:"+ str(round(float(position[0]), 2)) +", " + str(round(float(position[1]), 2)) +", " + str(round(float(position[2]),2)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        frame_blur = cv2.medianBlur(frame, 15)
        frame_hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        my_mask = cv2.inRange(frame_hsv, lower_bound, upper_bound)
        contours, hierarchy = cv2.findContours(my_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contours1=[]
        cv2.line(frame, (240, 0), (240, dispH), (0, 255, 0), thickness=2)
        cv2.line(frame, (400, 0), (400, dispH), (0, 255, 0), thickness=2)
        x_distances = []
        for c in contours:
            x, y, width, height = cv2.boundingRect(c)
            if (y+(height/2)) > dispH/2 -600 and 400<(height*width)<15000: #ASSUME area=100x100
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 2)
                centre_distance = int(dispW / 2) - (x + width/2)
                if centre_distance>(-dispW/2) and centre_distance<(dispW/2):
                    x_distances.append([abs(centre_distance), centre_distance])
            else:
                pass
        x_distances.sort()
        if len(x_distances) > 0 and countdown>0:
            closest_distance = min(x_distances)
            threading.Thread(target=send_data, args=(ser, closest_distance)).start()
        elif countdown<0:
        cv2.imshow("Camera", frame)
        cv2.imshow('my Mask',my_mask)
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows()
    ser.close()


if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    pc = Picamera2()
    threading.Thread(target=process_image, args=(pc, ser,countdown)).start()

