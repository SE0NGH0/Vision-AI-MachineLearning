# https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html 
# 맷플롯은 RGB

import cv2
import matplotlib.pyplot as plt
#= from matplotlib import pyplot as plt

img_color = cv2.imread('images/person_2.jpg', cv2.IMREAD_COLOR)

plt.imshow(img_color) # 파란 그림됨. BGR을 RGB로 보여줘서.
cv2.imshow('color', img_color)  # 원래 그림.
plt.xticks([])  # x축 눈금 표시 안 함. 표시하려면 이거 제거.
plt.yticks([])  # y축 눈금 표시 안 함. 표시하려면 이거 제거.
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()