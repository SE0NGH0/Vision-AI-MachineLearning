import cv2
import matplotlib.pyplot as plt

img_color = cv2.imread('./images/person_2.jpg', cv2.IMREAD_COLOR)

# BGR -> RGB 변환 1
b, g, r = cv2.split(img_color)    # 파랑 사진
img_color2 = cv2.merge([r, g, b]) # 정상 사진
 
# BGR -> RGB 변환 2
# img_color2 = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

plt.imshow(img_color2)
plt.xticks([])
plt.yticks([])
plt.show()