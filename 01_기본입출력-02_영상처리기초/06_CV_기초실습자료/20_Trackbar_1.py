# https://opencv-python.readthedocs.io/en/latest/doc/05.trackBar/trackBar.html
# 트랙바 만들기 (Nothing 콜백함수)

import cv2
import numpy as np

# createTrackbar의 콜백함수
def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow("Trackbar")

# Trackbar창에 트랙바 생성
cv2.createTrackbar('B', 'Trackbar', 0, 255, nothing)

while True:
    # 트랙바 현재 값 가져오기
    blue_value = cv2.getTrackbarPos('B', 'Trackbar')
    # 트랙바 값에 따라 이미지 업데이트
    img[:,:,0] = blue_value

    cv2.imshow('Trackbar', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()