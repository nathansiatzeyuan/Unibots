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

circles = []
while True:
    
    frame = pc.capture_array()
    frameBlur = frame
   # frameBlur= cv2.blur(frameBlur, (15,15))
    
    frameBlur = cv2.GaussianBlur(frameBlur, (11,11), 0)
    gray = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.7, 50)
    if circles is not None:
        circles = np.round(circles[0,:].astype("int"))
        for (x, y, r) in circles:
            
            cv2.circle(frame, (x, y), r, (0,255,0), 3)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()