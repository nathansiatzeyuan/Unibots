from picamera2 import Picamera2
import time
import cv2
pc = Picamera2()

pc.preview_configuration.main.size=(1280, 720)
pc.preview_configuration.main.format="RGB888"
pc.preview_configuration.align()
pc.configure("preview")
pc.start()


while True:
    frame = pc.capture_array()
    cv2.imshow("camera feed", frame)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
