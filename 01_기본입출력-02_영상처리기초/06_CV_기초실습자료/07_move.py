# https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
# 이미지를 2D 평행 이동(Translation) 시키는 예제
# 평행 이동은 이미지를 x축 또는 y축 방향으로 이동시키는 변환을 말합니다. 이 코드는 이미지가 오른쪽으로 40픽셀, 아래로 30픽셀 이동되도록 설정

import cv2
import numpy as np

# 1. 이미지를 읽기
img = cv2.imread('images/person_1.png')  # 이미지를 읽어옴

# 2. 이미지의 행(row)과 열(col) 크기 가져오기
row, col = img.shape[:2]  # img.shape[:2]: 이미지의 높이(row)와 너비(col) 가져오기

# 3. 이동 변환 행렬 정의
#    - [1, 0, 40]: x축 방향으로 40픽셀 이동
#    - [0, 1, 30]: y축 방향으로 30픽셀 이동
mat = np.float32([[1, 0, 40], [0, 1, 30]])

# 4. 이미지에 이동 변환 적용
#    - warpAffine(): 지정된 행렬(mat)에 따라 이미지를 변환
#    - (col, row): 결과 이미지의 크기 설정 (원본 크기 유지)
dst = cv2.warpAffine(img, mat, (col, row))

# 5. 원본 이미지 출력
cv2.imshow('Source', img)

# 6. 이동 변환된 이미지 출력
cv2.imshow('Translation', dst)

# 7. 키 입력 대기 및 모든 창 닫기
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
