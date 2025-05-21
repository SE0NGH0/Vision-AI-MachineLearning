# https://opencv-python.readthedocs.io/en/latest/doc/07.imageArithmetic/imageArithmetic.html#id3
# 이미지 합성: 두 이미지를 합성하여 하나의 이미지에 삽입합니다. 
# - 특정 이미지를 배경 이미지 위에 삽입하며, 
# - 삽입 이미지의 투명 배경 처리(mask)를 활용하여 자연스럽
# - 로고 이미지의 투명 배경 처리(mask)를 통해 자연스럽게 배경 이미지 위에 로고를 삽입하는 작업을 수행게 합성합니다.


import cv2
import numpy as np

# 1. 이미지 읽기
img1 = cv2.imread('./images/person_1.png')  # 배경 이미지
img2 = cv2.imread('./images/logo_2.png')   # 삽입할 로고 이미지

# 2. 삽입할 로고 이미지를 100x100 크기로 조정
img2 = cv2.resize(img2, (100, 100))

# 3. 삽입할 이미지의 크기 (row, col, channel) 정보 가져오기
rows, cols, channels = img2.shape

# 4. 배경 이미지에서 삽입 영역(ROI: Region of Interest) 지정
#    ROI: 배경 이미지의 (0, 0)부터 삽입할 이미지 크기(rows, cols)만큼 지정
roi = img1[0:rows, 0:cols]

# 5. 삽입할 로고 이미지(img2)를 그레이스케일로 변환 (흑백 이미지)
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 6. 그레이스케일 이미지를 이진화 (binary 이미지)로 변환
#    mask: 흰색(255) = 로고, 검정(0) = 배경
ret, mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)

# 7. mask를 반전하여 로고가 검정(0), 배경이 흰색(255)인 mask_inv 생성
mask_inv = cv2.bitwise_not(mask)

# 8. bitwise_and()로 배경 제거
#    - mask: 로고 부분만 남기고 배경 제거
#    - img2_logo: 로고 이미지만 남김
img2_logo = cv2.bitwise_and(img2, img2, mask=mask)

# 9. 배경 이미지(roi)에서 로고 삽입 영역을 비움
#    - mask_inv: 로고 삽입할 영역의 배경만 남김
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# 10. 두 이미지를 합성
#     - 로고 부분(img2_logo)과 배경 부분(img1_bg)을 합쳐서 dst 생성
dst = cv2.add(img2_logo, img1_bg)

# 11. 합성한 이미지를 배경 이미지에 삽입
#     - img1의 해당 ROI 영역에 dst 삽입
img1[0:rows, 0:cols] = dst

# 12. 결과 이미지 출력
cv2.imshow('Result', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
