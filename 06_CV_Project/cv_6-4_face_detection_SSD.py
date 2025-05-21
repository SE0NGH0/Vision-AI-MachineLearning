import sys
import numpy as np
import cv2

## SSD Face detector tensorflow 모델
# 참고: 모델 다운로드 링크
# https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/opencv_face_detector.pbtxt
# https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/opencv_face_detector_uint8.pb
model = 'opencv_face_detector_uint8.pb'  # 가중치 파일
config = 'opencv_face_detector.pbtxt'     # 네트워크 구성 파일

# 1. 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('카메라 열기 실패!')
    sys.exit()

# 2. DNN 모델 로드
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('모델 로드 실패!')
    sys.exit()

# 3. 메인 루프
while True:
    # 3-1. 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 3-2. 입력 프레임을 blob 형태로 변환
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))

    # 3-3. blob을 네트워크 입력으로 설정
    net.setInput(blob)

    # 3-4. 네트워크 순전파(forward) 실행하여 결과 얻기
    out = net.forward()  # out.shape = (1, 1, 200, 7)

    detect = out[0, 0, :, :]  # 검출 결과 가져오기
    (h, w) = frame.shape[:2]  # 프레임 크기 저장

    # 3-5. 검출 결과 반복 처리
    for i in range(detect.shape[0]):
        confidence = detect[i, 2]  # 신뢰도 (confidence)

        if confidence > 0.5:  # 신뢰도가 50% 넘으면 얼굴로 판단
            # detect 배열에서 좌상단(x1,y1), 우하단(x2,y2) 좌표 가져오기
            x1 = int(detect[i, 3] * w)
            y1 = int(detect[i, 4] * h)
            x2 = int(detect[i, 5] * w)
            y2 = int(detect[i, 6] * h)

            # 3-6. 얼굴 주위에 사각형 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

            # 3-7. 신뢰도 표시
            label = f'Face: {confidence:4.2f}'
            cv2.putText(frame, label, (x1, y1 - 1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    # 3-8. 결과 화면에 출력
    cv2.imshow('frame', frame)

    # 3-9. 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 4. 종료 처리
cap.release()
cv2.destroyAllWindows()
