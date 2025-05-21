# https://opencv-python.readthedocs.io/en/latest/doc/05.trackBar/trackBar.html
# 트랙바 만들기

import cv2
import numpy as np

def nothing(x):
    pass

# 창 하나 만들기
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Trackbar')

# 트랙바를 생성하여 창에 등록
cv2.createTrackbar('R', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('G', 'Trackbar', 0, 255, nothing)
cv2.createTrackbar('B', 'Trackbar', 0, 255, nothing)

switch = '0:Off\n1:On'
cv2.createTrackbar(switch, 'Trackbar', 1, 1, nothing)

while(1):
    cv2.imshow('Trackbar', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    r = cv2.getTrackbarPos("R", 'Trackbar')
    g = cv2.getTrackbarPos("G", 'Trackbar')
    b = cv2.getTrackbarPos('B', 'Trackbar')
    s = cv2.getTrackbarPos(switch, 'Trackbar')

    if s == 0:
        img[:] = 0 #모든 행/열 좌표값 = 0. 검은색
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()
