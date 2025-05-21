#########################################################################################3##############################
# 미션 4:
# 이미지를 HSV 색상 공간으로 변환하여 특정 색상(파란색)을 추출하고, 이를 마스크를 이용해 필터링한 결과를 출력하는 프로그램입니다.
#    - images/person_4.jpg 이미지를 HSV 색상 공간으로 변환하세요.
#    - HSV 색상 공간에서 파란색 영역을 필터링하여 마스크를 생성하세요.
#    - 파란색 영역만 남기고 나머지 색상은 제거된 결과 이미지를 생성하세요.
#    - 결과:
#        - 원본 이미지, HSV 이미지, 생성된 마스크, 그리고 파란색 영역만 추출된 이미지를 화면에 출력.
#    - 조건:
#        - HSV 색공간에서 파란색 범위는 H: 110~130, S: 50~255, V: 50~255.
#    - 주의:
#        - 생성된 마스크와 필터링된 이미지를 정확히 적용하여 결과를 출력하세요.
########################################################################################################################

import cv2
import numpy as np

# 1. 이미지 읽기
img = cv2.imread('../images/person_4.jpg')
if img is None:
    raise FileNotFoundError("images/person_4.jpg를 찾을 수 없습니다.")

# 2. BGR → HSV 변환
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. 파란색 영역 필터링을 위한 범위 지정 (H:110~130, S:50~255, V:50~255)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# 4. 마스크 생성 (파란색 픽셀만 흰색으로)
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 5. 마스크를 이용해 원본 이미지에서 파란색 영역만 추출
filtered = cv2.bitwise_and(img, img, mask=mask)

# 6. 결과 출력
cv2.imshow('Original Image', img)
cv2.imshow('HSV Image', hsv)
cv2.imshow('Blue Mask', mask)
cv2.imshow('Filtered (Blue Only)', filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
