# https://opencv-python.readthedocs.io/en/latest/doc/06.operation/operation.html
# 이미지 Channels

import cv2
import numpy as np

img = cv2.imread('images/person_4.jpg')

# BGR -> RGB로 변환
b, g, r = cv2.split(img)
img2 = cv2.merge((r,g,b))
img3 = img[:, :, 0]        #[모든 행,모든 열,첫째 채널] 0=Blue, 1=Green, 2=Red
cv2.imshow('img3', img3)

# Red만 제거
img[:,:,2] = 0 # Red Channel = 0
cv2.imshow('img4', img)

# Blue만 남기고 나머지 제거
img[:,:,1] = 0  # Blue = 0
img[:,:,2] = 0  # Red = 0
cv2.imshow('img5', img)

cv2.waitKey(0)
cv2.destroyAllWindows()