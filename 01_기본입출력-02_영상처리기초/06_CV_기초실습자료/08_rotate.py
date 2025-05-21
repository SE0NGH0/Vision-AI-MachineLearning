# https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html
# 이미지를 회전(Rotation) 및 **축소(Scaling)**시키는 예제

import cv2

# 1. 이미지를 읽어오기
img = cv2.imread(r'images\person_1.png')  # 이미지를 읽어와 img 변수에 저장

# 2. 이미지의 행(row)과 열(col) 크기 가져오기
row, col = img.shape[:2]  # img.shape[:2]: 높이(row)와 너비(col) 추출

# 3. 회전 변환 행렬 생성
#    - (col/2, row/2): 이미지 중심을 기준으로 회전
#    - 90: 회전 각도(90도 시계 방향 회전)
#    - 0.5: 크기 축소 비율(50%)
mat = cv2.getRotationMatrix2D((col/2, row/2), 90, 0.5)

# 4. 회전 변환 적용
#    - warpAffine(): 지정된 행렬(mat)을 사용하여 이미지를 변환
#    - (col, row): 결과 이미지 크기 설정
dst = cv2.warpAffine(img, mat, (col, row))

# 5. 원본 이미지 출력
cv2.imshow('Source', img)

# 6. 회전 및 축소된 이미지 출력
cv2.imshow('Rotated', dst)

# 7. 키 입력 대기 및 모든 창 닫기
cv2.waitKey(0)  # 키 입력 대기
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
