# https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html 
# 타원 그리기

import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
img = cv2.ellipse(img, (100,100), (100,50), 45, 0, 270, 255, -1)

cv2.imshow('Ellipse', img)
cv2.waitKey(0)
cv2.destroyAllWindows()