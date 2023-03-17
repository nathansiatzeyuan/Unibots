import apriltag
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True)
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


print("got image")
options = apriltag.DetectorOptions(families ="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[] aprtags detected", format(len(results)))

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
    print(tagFamily)
    
cv2.imshow("Image", image)
cv2.waitKey(0)