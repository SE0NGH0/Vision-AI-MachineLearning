# https://opencv-python.readthedocs.io/en/latest/doc/07.imageArithmetic/imageArithmetic.html
# 이미지 연산: add()로 이미지 더하기 

import cv2
import numpy as np

img1 = cv2.imread('images/flower1.jpg')
img2 = cv2.imread('images/flower2.jpg')

img_add = cv2.add(img1, img2)

cv2.imshow("Mix", img_add)

cv2.waitKey(0)
cv2.destroyAllWindows()