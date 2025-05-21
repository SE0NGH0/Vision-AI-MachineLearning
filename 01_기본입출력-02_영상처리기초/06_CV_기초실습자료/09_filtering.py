# https://opencv-python.readthedocs.io/en/latest/doc/11.imageSmoothing/imageSmoothing.html
# OpenCV와 트랙바(Trackbar)를 사용하여 이미지를 커널 크기에 따라 필터링(블러 효과)하는 과정을 실시간으로 조정하는 예제

import cv2
import numpy as np

# 1. 트랙바의 콜백 함수 정의
#    - 트랙바와 연동된 콜백 함수로 동작은 필요 없으므로 pass 처리
def nothing(x):
    pass

# 2. 이미지를 읽기
img = cv2.imread('images/person_1.png')  # 이미지를 읽어옴

# 3. OpenCV 창 생성 및 트랙바 추가
cv2.namedWindow('Filter')  # 'Filter'라는 창 생성
cv2.createTrackbar('F', 'Filter', 1, 20, nothing)  # 'F' 트랙바 추가, 초기값=1, 최대값=20

# 4. 무한 루프로 실시간 트랙바 값 확인 및 필터 적용
while True:
    # 4-1. ESC 키(코드 27) 입력 시 루프 종료
    if cv2.waitKey(1) & 0xff == 27:
        break

    # 4-2. 트랙바에서 현재 값 가져오기
    fil = cv2.getTrackbarPos('F', 'Filter')

    # 4-3. 필터 크기가 0이면 1로 설정 (0x0 커널은 허용되지 않음)
    if fil == 0:
        fil = 1

    # 4-4. 필터 커널 생성
    #      - 커널 크기: fil x fil
    #      - 값: 1/(fil*fil)로 나누어 평균값 필터 역할 수행
    kernel = np.ones((fil, fil), np.float32) / (fil * fil)

    # 4-5. 필터 적용
    #      - filter2D(): 커널을 이용해 이미지를 필터링
    dst = cv2.filter2D(img, -1, kernel)

    # 4-6. 필터링된 이미지 출력
    cv2.imshow('Filter', dst)

# 5. 키 입력 대기 및 모든 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
