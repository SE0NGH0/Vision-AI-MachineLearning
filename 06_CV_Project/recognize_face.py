# recognize_face.py
import cv2
import sys

def recognize_face(model_path: str = 'face_model.yml', threshold: float = 75.0):
    # 1) 분류기 로드 (OpenCV 내장 경로 사용)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"XML 파일을 로드할 수 없습니다: {cascade_path}")
        sys.exit(1)

    # 2) 학습된 LBPH 모델 로드
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(model_path)
    except cv2.error as e:
        print(f"모델 파일을 읽을 수 없습니다: {model_path}")
        sys.exit(1)

    # 3) 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 인덱스 0을 확인하세요.")
        sys.exit(1)

    print("얼굴 인식 시작 ('q' 키로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # (선택) 좌우 반전
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, confidence = recognizer.predict(roi)

            if confidence < threshold:
                text, color = "ACCESS GRANTED", (0, 255, 0)
            else:
                text, color = "ACCESS DENIED", (0, 0, 255)

            cv2.putText(
                frame,
                f"{text} ({confidence:.1f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_face()
