from imutils import perspective
import cv2
import numpy as np
import imutils

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

image = cv2.imread(r"C:\Users\urvil solanki\Downloads\image_assignment.jpg")
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

pixels_per_mm = None

for c in cnts:
    if cv2.contourArea(c) < 100:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    dA = np.linalg.norm(np.array([tltrX, tltrY]) - np.array([blbrX, blbrY]))
    dB = np.linalg.norm(np.array([tl[0], tl[1]]) - np.array([tr[0], tr[1]]))

    if pixels_per_mm is None:
        pixels_per_mm = dB / 88.0  

    dimA = dA / pixels_per_mm
    dimB = dB / pixels_per_mm

    print(f"Object dimensions: {dimA:.1f}mm x {dimB:.1f}mm")

    cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.putText(image, f"{dimA:.1f}mm x {dimB:.1f}mm", (int(tl[0]), int(tl[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
