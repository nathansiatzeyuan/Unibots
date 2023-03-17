import numpy as np
import time
from pupil_apriltags import Detector
import apriltag
import cv2
from picamera2 import Picamera2
pc=Picamera2()
res = (640, 320)
pc.preview_configuration.main.size= res
pc.preview_configuration.main.format="YUV420"
pc.preview_configuration.align()
pc.preview_configuration.controls.FrameRate=25
pc.configure("preview")
pc.start()

while True:
    tstart=time.time()
    image = pc.capture_array()
    gray = np.frombuffer(image,dtype=np.uint8,count=res[0] * res[1]).reshape(res[1], res[0])
    options = apriltag.DetectorOptions(families ="tag36h11", )
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    for r in results:
        (A, B, C, D) = r.corners
        A = (int(A[0]), int(A[1]))
        B = (int(B[0]), int(B[1]))
        C = (int(C[0]), int(C[1]))
        D = (int(D[0]), int(D[1]))
    
        cv2.line(image, A, B, (0, 255, 0), 2)
        cv2.line(image, B, C, (0, 255, 0), 2)
        cv2.line(image, C, D, (0, 255, 0), 2)
        cv2.line(image, A, D, (0, 255, 0), 2)
                 
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
        
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (A[0], A[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        
    
    
    cv2.imshow("cam", image)
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()