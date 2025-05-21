# https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
# **이미지를 이진화(Thresholding)**하는 다양한 방법을 비교하여 결과를 시각화

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. 이미지를 그레이스케일로 읽기
img = cv2.imread('images/person_1.png', 0)  # 0: Grayscale로 읽기

# 2. (선택사항) 이미지를 Median Blur로 부드럽게 처리 (주석 처리됨)
# img = cv2.medianBlur(img, 5)

# 3. 글로벌 이진화 적용
#    - 임계값 127을 기준으로, 127 이상은 흰색(255), 미만은 검정(0)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 4. Adaptive Thresholding (Mean 방식) 적용
#    - 주변 픽셀의 평균값을 기준으로 이진화
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

# 5. Adaptive Thresholding (Gaussian 방식) 적용
#    - 주변 픽셀의 가중 평균값(가우시안 가중치)을 기준으로 이진화
thresh3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

# 6. 결과를 시각적으로 비교하기 위한 제목과 이미지 리스트 정의
titles = ['Source', 'Global', 'Mean', 'Gaussian']  # 제목: 원본, 글로벌, 평균, 가우시안
images = [img, thresh1, thresh2, thresh3]  # 비교할 이미지 리스트

# 7. Matplotlib으로 이미지 출력
for i in range(4):
    plt.subplot(2, 2, i+1)  # 2x2 형태의 서브플롯 생성
    plt.imshow(images[i], 'gray')  # 이미지를 흑백으로 출력
    plt.title(titles[i])  # 각 서브플롯에 제목 추가
    plt.xticks([]), plt.yticks([])  # x축, y축 눈금 제거

# 8. 결과 출력
plt.show()
