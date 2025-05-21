#  OpenCV와 Haar Cascade를 사용하여 실시간으로 웹캠에서 얼굴을 검출하고, 검출된 얼굴 영역을 모자이크 처리하여 화면에 표시하는 프로그램

import cv2
import numpy as np

# 1. 카메라 객체 생성 및 해상도 설정
cap = cv2.VideoCapture(0)  # 0번 카메라 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# 2. Haar Cascade 검출기 객체 생성
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 얼굴 검출기
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # 눈 검출기 (옵션)

# 3. 실행 루프
while True:
    # 3-1. 현재 프레임 읽기
    ret, frame = cap.read()  # 현재 프레임 읽기
    if not ret:  # 프레임 읽기 실패 시 루프 종료
        break

    # 3-2. 회색조 변환 (Haar Cascade는 흑백 이미지에서 동작)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3-3. 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 3-4. 얼굴 영역 모자이크 처리
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_region = frame[y:y+h, x:x+w]

        # 모자이크 적용 (이미지를 축소한 뒤 확대)
        mosaic_level = 15  # 모자이크 강도 (숫자가 클수록 더 강한 모자이크)
        small_face = cv2.resize(face_region, (w // mosaic_level, h // mosaic_level), interpolation=cv2.INTER_LINEAR)
        mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)

        # 원본 이미지에 모자이크 덮어쓰기
        frame[y:y+h, x:x+w] = mosaic_face

    # 3-5. 결과 영상 출력
    cv2.imshow("Mosaic Face Detection", frame)

    # 3-6. 'q' 키 입력 시 루프 종료
    if cv2.waitKey(1) == ord('q'):  # 1ms 대기 후 키 입력 확인
        break

# 4. 종료 처리
cap.release()  # 카메라 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
