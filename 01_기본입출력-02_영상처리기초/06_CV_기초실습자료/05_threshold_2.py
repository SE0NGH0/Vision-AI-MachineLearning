# https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html
# 예) 임계값보다 작으면 = 0, 크면 = 255

import cv2
import numpy as np
from matplotlib  import pyplot as plt

img = cv2.imread("images/person_4.jpg")

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Sorce', 'Binary', 'Binary inv', 'Trunc', 'To zero', 'To Zero Inv']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    
plt.show()
