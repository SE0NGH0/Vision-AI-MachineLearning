# Otsu 이진화: 전경과 배경을 가장 잘 구분할 수 있는 임계값을 자동으로 계산함.

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/person_2.jpg', cv2.IMREAD_GRAYSCALE)

_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(otsu, cmap='gray')
plt.title("Otsu")
plt.show()