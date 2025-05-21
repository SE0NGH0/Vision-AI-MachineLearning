# https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
# 이미지 보간법 비교 (내 눈에는 다 같아보임)

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/person_1.png')
height, width = img.shape[:2]

# 보간법: INTER_AREA
img_area = cv2.resize(img, (width*2, height*2), interpolation= cv2.INTER_AREA)

# 보간법: INTER_LINEAR 
img_linear = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# 보간법: INTER_CUBIC 
img_cubic = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC)

# 보간법: INTER_NEAREST 
img_nearest = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

images = [img, img_area, img_linear, img_cubic, img_nearest]
titles = ['source', 'area', 'linear', 'cubic', 'nearest']

for i in range(5):
    cv2.imshow(titles[i], images[i])
cv2.waitKey(0)
cv2.destroyAllWindows()


# for i in range(5):
#     plt.subplot(2,3,i+1)
#     plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
