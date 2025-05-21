# 이미지를 그림 스타일로 변형하기.
# python -m pip install matplotlib

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 읽기 : 방법1 - cv2.imread() : 경로에 한글/특수문자가 없는 경우
# img = cv2.imread('./Images/3333.jpg', cv2.IMREAD_UNCHANGED)

# 이미지 읽기 : 방법2 - np.fromfile() + cv2.imdecode()
# 이미지 파일을 바이트 배열로 읽기
#ff = np.fromfile(r'\images\3333.jpg', np.uint8)
ff = np.fromfile('./images/3333.jpg', np.uint8)

# 바이트 배열을 이미지로 디코딩 (원본이미지를 그대로(unchanged) 읽어옴)
img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

# 이미지를 크기 조정
img = cv2.resize(img, dsize=(0,0), fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR) 
## dsize(0,0): fx, fy 값대로 사이즈 조정
## fx=1.0, fy=1.0: 원본과 같은 크기 x, y로
## Interpolation: 이미지 사이즈 변경 옵션(픽셀값 계산법 옵션)

# 이미지를 스타일화
pic = cv2.stylization(img, sigma_s=100, sigma_r=0.1)
## sigma_s: 공간필터의 표준편차. 클수록 넓은 영역 고려해 스타일 적용
## sigma_r: 색상필터의 표준편차. 작을수록 세밀한 색상변화 허용. 0.1=세밀

# 이미지 표시 (matplotlib 사용)
plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
plt.title('Painting Style')
plt.axis('off')  # 축 표시하지 않음
plt.show()
