# https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
# 이미지 리사이징

import cv2
import numpy as np

img_src = cv2.imread('images/person_1.png')

img_re1 = cv2.resize(img_src, (300, 400))
img_re2 = cv2.resize(img_src, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imshow('Source', img_src)
cv2.imshow('Re1', img_re1)
cv2.imshow('Re2', img_re2)

cv2.waitKey(0)
cv2.destroyAllWindows()