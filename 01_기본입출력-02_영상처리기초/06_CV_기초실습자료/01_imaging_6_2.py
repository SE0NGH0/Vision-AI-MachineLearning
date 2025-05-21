# 미션: 트랙바를 달아 이미지 수동 조절해보기. (트랙바 배운 후 진행할것)
# OpenCV 문제로 동작 안 될 경우 6_2로 실행해보자.

import numpy as np
import cv2
import matplotlib.pyplot as plt  # python -m pip install matplotlib

# 1. 이미지 읽기
# 이미지를 바이트 배열로 읽기
# ff = np.fromfile(r'images\1111.jpg', np.uint8)
# # 바이트 배열을 이미지로 디코딩(원본 이미지를 그대로(unchanged) 읽어옴)
# img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)

img = cv2.imread(r'images\1111.jpg', cv2.IMREAD_UNCHANGED )

# 2. 이미지 크기 조정 : fx=1.0, fy=1.0 이미지 크기 변경 안됨
img = cv2.resize(img, dsize=(0, 0), fx=1.0, fy=1.0, interpolation=cv2.INTER_LINEAR)

def changeCallback(pos):
    pass

# 3. 트랙바 생성
cv2.namedWindow("Painting Trackbar")
cv2.createTrackbar("sigma_s", "Painting Trackbar", 0, 200, changeCallback)
cv2.createTrackbar("sigma_r", "Painting Trackbar", 0, 100, changeCallback)

# 3-1. 트랙바 초깃값 설정
cv2.setTrackbarPos("sigma_s", "Painting Trackbar", 100)
cv2.setTrackbarPos("sigma_r", "Painting Trackbar", 10)

while True:
    if cv2.waitKey(100) == ord('q'):
        break

    sigma_s_value = cv2.getTrackbarPos("sigma_s", "Painting Trackbar")
    sigma_r_value = cv2.getTrackbarPos("sigma_r", "Painting Trackbar") / 100.0

    print("sigma_s_value: ", sigma_s_value)
    print("sigma_r_value: ", sigma_r_value)

    # 4. 스타일화 적용 : 페인팅 스타일로 변환
    pic = cv2.stylization(img, sigma_s=sigma_s_value, sigma_r=sigma_r_value)
    ## sigma_s: 공간 필터의 표준편차. 클수록 넓은 영역을 고려해 스타일 적용
    ## sigma_r: 색상 필터의 표준편차. 작을수록 세밀한 색상 변화를 허용. 0.1=세밀

    cv2.imshow("Painting Style", pic)

# Matplotlib 출력 (루프 종료 후 한 번만 표시)
plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
plt.title("Painting Style")
plt.axis('off')  # 축 표시하지 않음
plt.show()
#plt.pause(0.1)  # 약간의 지연을 추가하여 창이 제대로 업데이트되도록 함

cv2.destroyAllWindows()
