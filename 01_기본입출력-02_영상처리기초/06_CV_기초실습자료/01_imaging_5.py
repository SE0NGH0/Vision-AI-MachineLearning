# https://opencv-python.readthedocs.io/en/latest/doc/07.imageArithmetic/imageArithmetic.html
# 이미지 연산: Trackbar로 이미지 더하기 2

import cv2
import numpy as np

img1 = cv2.imread('./images/person_1.png')
img2 = cv2.imread('./images/flower2.jpg')

# 이미지 크기 조정 (img2 크기를 img1 크기로 맞추기)
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

def nothing(x):
    pass

cv2.namedWindow('Blender')
cv2.createTrackbar('W', 'Blender', 0, 100, nothing)
# Blender라는 창에 "W"라는 트랙바를 생성(0 ~ 100), 콜백함수=nothing

while True:
    # 슬라이더의 현재값을 pos에 대입
    pos = cv2.getTrackbarPos('W', 'Blender')

    # img1, img2에 대한 가중합
    w1 = pos / 100.0   # (0 ~ 100)/100
    w2 = 1.0 - w1
    dst = cv2.addWeighted(img1, w1, img2_resized, w2, 0)  # cv2.addWeighted(src1, alpha, src2, beta, gamma)

    cv2.imshow("Blender", dst)

    if cv2.waitKey(1) & 0xff == 27:  # 'esc' key
        break

cv2.destroyAllWindows()