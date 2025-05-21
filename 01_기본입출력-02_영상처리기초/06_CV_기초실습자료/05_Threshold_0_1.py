# 이미지를 이진화합니다.

import cv2
import matplotlib.pyplot as plt

img_color = cv2.imread("images/person_4.jpg", 0)
ret, img_binary = cv2.threshold(img_color, 127, 255, cv2.THRESH_BINARY)

plt.imshow(img_binary, cmap='gray')
plt.show()