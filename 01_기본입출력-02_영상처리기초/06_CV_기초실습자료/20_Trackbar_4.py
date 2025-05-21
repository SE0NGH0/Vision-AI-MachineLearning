import cv2
import numpy as np

img1 = np.zeros((300,300,3), np.uint8)     #검정 배경
img2 = np.zeros((300,300,3), np.uint8) * 255 #흰 배경

def nothing(x):
    pass

cv2.namedWindow('Blender')

# Trackbar 생성
cv2.createTrackbar('bg', 'Blender', 0, 255, nothing)

while True:
    # 트랙바의 위치에 따라 img1, img2 색상 변환
    bg = cv2.getTrackbarPos('bg', 'Blender')

    img1[:, :, 0] = 255 - bg
    img2[:, :, 1] = bg
    
    img_add = cv2.add(img1, img2)

    cv2.imshow('Blender', img_add)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
