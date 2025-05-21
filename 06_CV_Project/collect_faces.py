import cv2
import os


def collect_faces(user_id: str, max_count: int = 100, output_dir: str = 'dataset'):
    # Haar Cascade XML 경로 설정
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"XML 파일을 로드할 수 없습니다: {cascade_path}")

    # 카메라 고정 인덱스 0 및 해상도 설정
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("카메라를 열 수 없습니다. 인덱스 0에 연결된 카메라를 확인하세요.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("카메라 인덱스 0에서 열렸습니다.")

    # 저장 디렉토리 생성
    user_dir = os.path.join(output_dir, user_id)
    os.makedirs(user_dir, exist_ok=True)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 검출된 얼굴 저장 및 화면 표시
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            img_path = os.path.join(user_dir, f"{user_id}_{count:03d}.jpg")
            cv2.imwrite(img_path, face_resized)
            count += 1

            # 바운딩 박스 및 카운트 텍스트
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Count: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow('Face Collection', frame)

        # 종료 조건: 'q' 키 또는 최대 개수 도달
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= max_count:
            if count >= max_count:
                print(f"{max_count}장 수집 완료.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    uid = input("저장할 사용자 ID를 입력하세요: ")
    collect_faces(uid)
