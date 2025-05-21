# https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html 
# 다각형 그리기

import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
pts = np.array([[0,0],[70,50],[80,80],[30,70]], np.int32)
img = cv2.polylines(img, [pts], True, (0,0,255), 3)

cv2.imshow("Poly", img)
cv2.waitKey(0)
cv2.destroyAllWindows()