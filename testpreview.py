from picamera2 import Picamera2
import time

pc = Picamera2()
pc.start_and_capture_video("3tags")