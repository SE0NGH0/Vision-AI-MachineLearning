##########################################################################
## 미션 1 
## - 적당한 이미지를 다운받아 3가지 플래그를 적용하고 3개의 창에 띄우시오.
## - 저장한 이미지의 shape 을 출력하세요
## - 회색조 이미지를 저장하세요
## - 키보드 입력이 있을 때까지 대기하도록 하세요
## - 열린 모든 창을 닫으세요.
##########################################################################

import cv2

# 1. 이미지 파일 경로
img_path = '..\cv_images\img.jpg'  # 작업 디렉토리에 적당한 이미지 파일을 미리 내려받아 두세요

# 2. 세 가지 플래그로 이미지 읽기
img_color     = cv2.imread(img_path, cv2.IMREAD_COLOR)      # 컬러로 읽기
img_gray      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 읽기
img_unchanged = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함 등 원본 그대로 읽기

# 3. 읽기 성공 확인
if img_color is None or img_gray is None or img_unchanged is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

# 4. 각 창에 표시
cv2.imshow('COLOR (IMREAD_COLOR)',     img_color)
cv2.imshow('GRAYSCALE (IMREAD_GRAYSCALE)', img_gray)
cv2.imshow('UNCHANGED (IMREAD_UNCHANGED)', img_unchanged)

# 5. 그레이스케일 이미지 shape 출력
print('그레이스케일 이미지 shape:', img_gray.shape)

# 6. 그레이스케일 이미지 저장
cv2.imwrite('gray_saved.jpg', img_gray)
print('gray_saved.jpg 로 저장되었습니다.')

# 7. 키보드 입력 대기
cv2.waitKey(0)

# 8. 열린 모든 창 닫기
cv2.destroyAllWindows()
