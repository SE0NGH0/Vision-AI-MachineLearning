import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/person_4.jpg', 0)

ret, binary_image = cv2.threshold(img,127,255, cv2.THRESH_BINARY)

plt.subplot(1, 2, 1)   #1행 2열 중 1열 칸
plt.imshow(img, cmap='gray')
plt.title('Src Img')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)   #1행 2열 중 2열 칸
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.xticks([])
plt.yticks([])

plt.show()