# https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
# 글자 넣기

import cv2
import numpy as np

img = np.zeros((512,512,3), np.uint8)
img = cv2.rectangle(img, (0,0), (100,100), 255, 3)
img = cv2.putText(img, 'Acc: 100', (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

cv2.imshow('Text', img)
cv2.waitKey(0)
cv2.destroyAllWindows()