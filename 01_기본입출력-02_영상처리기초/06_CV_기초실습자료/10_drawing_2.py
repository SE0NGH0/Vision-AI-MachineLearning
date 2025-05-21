# https://opencv-python.readthedocs.io/en/latest/doc/03.drawShape/drawShape.html 
# 사각형 그리기

import cv2
import numpy as np

# 검정 배경 그리기
img = np.zeros((512,512,3), np.uint8)
# # 배경색 원하면 바꾸기
# img[:, :] = (100, 200, 200) #각 행렬 픽셀에 BGR를 대입함
# 사각형 그리기
img = cv2.rectangle(img, (0,0), (100,100),(0,255,0), 3)

cv2.imshow('Square', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
