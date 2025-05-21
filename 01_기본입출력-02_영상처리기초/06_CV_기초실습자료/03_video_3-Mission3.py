# OpenCV를 사용하여 웹캠에서 실시간 비디오를 캡처하고, 반전된 화면을 녹화하여 AVI 형식으로 저장

###################################################################################
# 미션 3: 
#    - 웹캠에서 실시간으로 비디오를 캡처하여 'images/capture.avi' 파일로 저장하세요
#    - 저장 조건:
#        - 해상도: 640x480
#        - 프레임 속도: 25 FPS
#        - 코덱: 'DIVX' 사용
#    - 캡처한 영상에 좌우 반전 효과를 적용하세요
#    - 'q' 키를 누르면 프로그램이 종료되도록 구현하세요
###################################################################################

import cv2
import os

# 저장 디렉토리 확인·생성
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

# 2. 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. 비디오 라이터 준비 (DIVX 코덱, 25FPS, 640x480)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(
    os.path.join(output_dir, 'capture.avi'),
    fourcc,
    25.0,
    (640, 480)
)

# 4. 프레임 읽기 & 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전
    flipped = cv2.flip(frame, 1)

    # 녹화
    out.write(flipped)

    # 화면에 표시
    cv2.imshow('Flipped Capture', flipped)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
