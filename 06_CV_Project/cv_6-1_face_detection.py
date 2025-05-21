# OpenCV와 Haar Cascade 검출기를 사용하여 웹캠으로 실시간 얼굴을 감지하고 사각형으로 표시하는 프로그램 - 얼굴/눈 인식
# HarrCascade 위치: pip show opencv-python으로 찾으면 됨. 

import cv2

# 1. 카메라 객체 생성 및 해상도 설정
capt = cv2.VideoCapture(0)  # 기본 웹캠 사용
capt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# 2. Harr Cascade 검출기 객체 생성 (OpenCV 내장 경로 사용)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 눈 검출기 (필요 시 활성화)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 3. 실행 루프
while True:
    # 3-1. 현재 프레임 읽기
    ret, frame = capt.read()

    # 3-2. 그레이스케일로 변환 (Haar Cascade는 흑백 이미지에서 동작)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3-3. 얼굴 검출 수행
    # scaleFactor: 이미지 축소 비율, minNeighbors: 인접 사각형 수, minSize: 최소 얼굴 크기
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3, minSize=(20, 20))

    # 3-4. 얼굴 검출 결과를 바운딩 박스로 표시
    if len(faces):  # 검출된 얼굴이 있으면
        for x, y, w, h in faces:  # 얼굴 좌표와 크기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_4)  # 파란색 사각형 그리기

    # 3-5. (선택) 눈 검출 기능 (필요 시 활성화)
    # eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3, minSize=(10, 10))
    # if len(eyes):  # 검출된 눈이 있으면
    #     for x, y, w, h in eyes:
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2, cv2.LINE_4)  # 노란색 사각형 그리기

    # 3-6. 좌우 반전 (웹캠 영상의 거울 효과)
    frame = cv2.flip(frame, 1)

    # 3-7. 결과 프레임 화면에 출력
    cv2.imshow("Original", frame)

    # 3-8. 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 4. 종료 처리
capt.release()  # 카메라 리소스 해제
cv2.destroyAllWindows()  # 모든 창 닫기
