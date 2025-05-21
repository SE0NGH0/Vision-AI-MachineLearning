import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 이미지를 바이트 배열로 읽기
ff = np.fromfile(r'images\1111.jpg', np.uint8)

# 바이트 배열을 이미지로 디코딩(원본 이미지를 그대로(unchanged) 읽어옴)
img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

# 이미지 크기 조정
img = cv2.resize(img, dsize=(0,0), fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

# 초기 sigma_s와 sigma_r 값
initial_sigma_s = 100
initial_sigma_r = 0.1

# 초기 스타일화 이미지 생성
pic = cv2.stylization(img, sigma_s=initial_sigma_s, sigma_r=initial_sigma_r)

# Matplotlib Figure 생성
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# 스타일화된 이미지 표시
image_display = plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 축 표시하지 않음
plt.title('Painting Style')

# Sigma_s 슬라이더 추가
ax_sigma_s = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_sigma_s = Slider(ax_sigma_s, 'Sigma_s', 0, 200, valinit=initial_sigma_s)

# Sigma_r 슬라이더 추가
ax_sigma_r = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_sigma_r = Slider(ax_sigma_r, 'Sigma_r', 0, 1, valinit=initial_sigma_r, valstep=0.01)

# 슬라이더 값이 변경될 때 호출되는 함수
def update(val):
    sigma_s = slider_sigma_s.val
    sigma_r = slider_sigma_r.val
    pic = cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)
    image_display.set_data(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    fig.canvas.draw_idle()

# 슬라이더 값이 변경될 때 update 함수 호출
slider_sigma_s.on_changed(update)
slider_sigma_r.on_changed(update)

plt.show()
