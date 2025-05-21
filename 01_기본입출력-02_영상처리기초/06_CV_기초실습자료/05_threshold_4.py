# 이미지 이진화(Thresholding) 방법의 차이를 시각적으로 비교하고, 각 방법의 히스토그램(픽셀 분포)을 함께 출력
# Otsu’s Thresholding과 Global Thresholding, 그리고 가우시안 필터링 후 Otsu’s Thresholding을 비교

import cv2
import matplotlib.pyplot as plt

# 1. 이미지를 그레이스케일로 읽기
img = cv2.imread('images/flower1.jpg', 0)

# 2. Global Thresholding 적용
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. Otsu's Thresholding 적용
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Gaussian Blur를 사용하여 노이즈 제거
blur = cv2.GaussianBlur(img, (5, 5), 0)

# 5. Gaussian Blur 후 Otsu's Thresholding 적용
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 6. 결과와 히스토그램 비교를 위한 데이터 및 제목 설정
images = [img, 0, th1, img, 0, th2, blur, 0, th3]
titles = ['Original Image', 'Histogram', 'Global Threshold',
          'Original Image', 'Histogram', "Otsu's Threshold",
          'Gaussian Blur', 'Histogram', "Gaussian + Otsu"]

# 7. Matplotlib으로 결과와 히스토그램 출력
for i in range(3):
    # 7-1. 이미지 출력
    plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3], fontsize=10)  # 폰트 크기를 10으로 설정
    plt.xticks([]), plt.yticks([])

    # 7-2. 히스토그램 출력 : 픽셀 값이 특정 범위에 몰려 있는지(밝거나(오르쪽치우침) 어두운(왼쪽치우침) 이미지) 또는 고르게 분포되어 있는지 확인
    plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1], fontsize=10)  # 폰트 크기를 10으로 설정
    plt.xticks([]), plt.yticks([])

    # 7-3. 이진화된 이미지 출력
    plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2], fontsize=10)  # 폰트 크기를 10으로 설정
    plt.xticks([]), plt.yticks([])

# 8. 결과 출력
plt.show()
