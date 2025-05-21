# https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html 
# 원 그리기

import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
img = cv2.circle(img, (70, 100), 70, (255,0,0), -1)

cv2.imshow('Circle', img)
cv2.waitKey(0)
cv2.destroyAllWindows()