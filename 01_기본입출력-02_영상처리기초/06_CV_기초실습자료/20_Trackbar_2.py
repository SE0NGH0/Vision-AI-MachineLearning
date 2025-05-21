# https://opencv-python.readthedocs.io/en/latest/doc/05.trackBar/trackBar.html
# 트랙바 만들기 (콜백함수로 파란 값 업데이트)

import cv2
import numpy as np

def trackbar_callback(x): #createTrackbar의 콜백함수
    img[:,:,0] = x  # 파란 채널 업데이트
    cv2.imshow('Trackbar', img)

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow("Trackbar")

# 트랙바 생성
cv2.createTrackbar('B', 'Trackbar', 0, 255, trackbar_callback)

while True:
    if cv2.waitKey(1) & 0xFF == 27: #Esc
        break

cv2.destroyAllWindows()
