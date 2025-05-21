###################################################################################
# 미션 2: 
#    - images/person_4.jpg 에서 하단의 공(ROI)을 복사하세요
#    - ROI: img[행 시작:끝, 열 시작:끝]
#    - 복사한 공을 본 이미지의 공 하단으로 복사하여 이미지에 공을 2개로 보이도록 하세요
#    - 주의: 행열의 폭이 일치해야함  
###################################################################################

import cv2

# 1. 이미지 읽기
img = cv2.imread('../images/person_4.jpg')
if img is None:
    raise FileNotFoundError("images/person_4.jpg를 찾을 수 없습니다.")

# 2. ROI 좌표 (행: y1~y2, 열: x1~x2) — 여러분 이미지에 맞게 조정하세요!
y1, y2 = 400, 460   # 예: 공의 세로 위치
x1, x2 = 830, 870   # 예: 공의 가로 위치

# 3. ROI 복사 (copy() 안 하면 원본 참조가 되므로 반드시 복사)
ball_roi = img[y1:y2, x1:x2].copy()

# 4. 붙여넣을 위치 계산
h, w = y2 - y1, x2 - x1
# 그대로 아래에 붙이려면 y2:y2+h, x1:x1+w
target_y1, target_y2 = y2, y2 + h
target_x1, target_x2 = x1, x1 + w

# 5. 이미지 경계 체크
rows, cols = img.shape[:2]
if target_y2 > rows or target_x2 > cols:
    raise ValueError("붙여넣을 영역이 이미지 크기를 벗어납니다. ROI 좌표를 조정하세요.")

# 6. ROI 붙여넣기
img[target_y1:target_y2, target_x1:target_x2] = ball_roi

# 7. 결과 표시
cv2.imshow('Duplicated Ball', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
