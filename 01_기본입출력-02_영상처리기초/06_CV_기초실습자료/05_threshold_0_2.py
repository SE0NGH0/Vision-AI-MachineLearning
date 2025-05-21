# Adaptive Thresholding을 사용하여 이미지의 밝기 변화를 동적으로 계산하고 특정 조건에 따라 이미지를 이진화(흑백)하는 예제 : 적응형 임계값
# 조명 변화나 그늘이 있는 이미지에서 객체를 잘 분리하기 위해 사용

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./images/person_2.jpg', cv2.IMREAD_GRAYSCALE)

# Adaptive 방법:
# - ADAPTIVE_THRESH_MEAN_C: 주변 픽셀의 평균 값을 기준으로 임계값을 계산.
# - (대안) ADAPTIVE_THRESH_GAUSSIAN_C: 주변 픽셀의 가중 평균 값을 기준으로 임계값 계산.
# - 11 (블록 크기): 임계값 계산에 사용할 주변 픽셀 영역의 크기 (11x11 블록).
# - 2 (C) : 평균값에서 뺄 상수. 밝기 보정에 사용됩니다.
adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

plt.imshow(adapt, cmap='gray')
plt.title("Adaptive Thresh")
plt.show()