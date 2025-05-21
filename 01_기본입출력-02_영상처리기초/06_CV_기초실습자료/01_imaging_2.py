# https://opencv-python.readthedocs.io/en/latest/doc/06.operation/operation.html
# 이미지 기본 조작

import cv2
import numpy as np

img = cv2.imread(r'images\person_1.png')

# 이미지 기본 속성(행, 열, 색채널)
# Grayscale이면 (행, 열)만 리턴
shape_ = img.shape
print(shape_)

#10행 20열 색값 확인
px = img[10,20]
print(px)   #[90, 92, 91]

#10행 20열 색값 중 특정 계열 색값 확인
blue_ = img[10, 20, 0]
green_ = img[10, 20, 1]
red_ = img[10, 20, 2]
print(blue_, green_, red_) #90 92 91

# #10행 20열 색값을 변경
img.item(10, 20, 2)  # 91
img.itemset((10,20,2), 255) # 91 -> 255
print(img.item(10,20,2))

# 전체 픽셀 수
print(img.size)

# 이미지 Datatype
print(img.dtype) # 0~255: uint8

