# https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html
# 선 그리기

import cv2
import numpy as np

# 검정배경: img=3차원(높512,넓512,3색(BRG)) 넘파이 배열
img = np.zeros((512,512,3),np.uint8) 
# 선 그리기: 이미지명, 시작좌표, 끝좌표, 선색, 선 두께
img= cv2.line(img, (0,0), (511,511), (255,0,0),5)

cv2.imshow('Draw', img)
cv2.waitKey(0)
cv2.destroyAllWindows()