# https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
# 이미지를 다양한 방법으로 크기를 조정(Resize)하고, 그 결과를 화면에 출력 - Resizing

import cv2
import numpy as np

# 1. 이미지를 읽어오기
img = cv2.imread('images/logo.png')  # 'logo.png' 이미지를 읽어옴

# 2. 이미지의 행(Height)과 열(Width) 가져오기
height, width = img.shape[:2]  # shape[:2]: 높이와 너비 가져오기

# 3. 비율로 이미지 크기 조정 (축소)
#    - fx=0.5, fy=0.5: 가로, 세로 모두 50%로 축소
#    - INTER_AREA: 축소 시 품질이 좋은 보간법 사용
shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# 4. 사용자 지정 크기로 이미지 크기 조정 (300x500)
#    - INTER_CUBIC: 확대 시 품질이 좋은 보간법 사용
custom1 = cv2.resize(img, (300, 500), interpolation=cv2.INTER_CUBIC)

# 5. 이미지 크기를 2배로 확대
#    - INTER_CUBIC: 확대 시 선명한 결과를 위한 보간법 사용
custom2 = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)

# 6. 원본 이미지 출력
cv2.imshow("Source", img)

# 7. 축소된 이미지 출력
cv2.imshow('Shrink', shrink)

# 8. 사용자 지정 크기의 이미지 출력 (300x500)
cv2.imshow('Custom1', custom1)

# 9. 2배로 확대된 이미지 출력
cv2.imshow('Custom2', custom2)

# 10. 키 입력 대기 및 모든 창 닫기
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
