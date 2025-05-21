# 웹캠을 통해 얼굴을 실시간으로 감지하고, 얼굴이 감지되면 자동으로 영상을 녹화하는 프로그램
# https://www.youtube.com/watch?v=2AE1OXij8z0&list=PLgkpDfSY2Bez0l8oyY0UNGusnk1ZFH7On&index=8

import cv2
from PIL import ImageFont, ImageDraw, Image  # 텍스트 추가와 그래픽 요소를 위한 라이브러리
import numpy as np
import datetime

# 1. 카메라 객체 생성 및 해상도 설정
capture = cv2.VideoCapture(0)  # 0번 카메라 사용
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# 2. 상태 변수 및 설정 초기화
is_record = False  # 녹화 여부 상태 변수 (처음에는 녹화 중이 아님)
on_record = False  # 녹화 이벤트 상태 변수
cnt_record = 0  # 현재 녹화 시간
max_cnt_record = 5  # 최소 촬영 시간 (단위: 초)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 녹화 파일 코덱 설정
font = ImageFont.truetype('fonts/SCDream6.otf', 20)  # 텍스트 표시를 위한 폰트 로드

# 3. Haar Cascade 검출기 객체 생성
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')  # (옵션) 눈 검출기

# 4. 실행 루프
while True:
    # 4-1. 현재 시간 저장 (텍스트와 파일명에 활용)
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')  # 화면 표시용 시간
    nowDatetime_path = now.strftime('%Y-%m-%d %H_%M_%S')  # 파일명용 시간 (특수문자 제거)

    # 4-2. 현재 프레임 읽기 및 흑백 변환
    ret, frame = capture.read()  # 현재 프레임을 읽음
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Haar Cascade는 흑백 이미지에서 동작하므로 변환

    # 4-3. 텍스트 배경 추가 및 현재 시간 표시
    cv2.rectangle(img=frame, pt1=(10, 15), pt2=(340, 35), color=(0, 0, 0), thickness=-1)  # 검은 배경
    frame = Image.fromarray(frame)  # Numpy 배열 이미지를 Pillow 이미지로 변환
    draw = ImageDraw.Draw(frame)  # Draw 객체 생성
    draw.text(xy=(10, 15), text="카이로스 웹캠 " + nowDatetime, font=font, fill=(255, 255, 255))  # 텍스트 추가
    frame = np.array(frame)  # Pillow 이미지를 다시 Numpy 배열로 변환

    # 4-4. 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3, minSize=(20, 20))

    # 4-5. 얼굴 검출 시 녹화 시작
    if len(faces):
        is_record = True  # 녹화 상태로 전환
        if not on_record:  # 녹화 이벤트가 초기화 상태라면
            video = cv2.VideoWriter("Capture/카이로스 웹캠 " + nowDatetime_path + ".avi", fourcc, 1, (frame.shape[1], frame.shape[0]))
        cnt_record = max_cnt_record  # 녹화 시간 초기화

    # 4-6. 녹화 중이면 프레임 저장
    if is_record:
        print('녹화 중')  # 상태 출력
        video.write(frame)  # 현재 프레임을 녹화 파일에 저장
        cnt_record -= 1  # 남은 녹화 시간을 감소
        on_record = True  # 녹화 이벤트 활성화

    # 4-7. 녹화 시간이 종료되면 녹화 상태 초기화
    if cnt_record == 0:
        is_record = False  # 녹화 상태 종료
        on_record = False  # 녹화 이벤트 종료

    # 4-8. 얼굴 검출 시 사각형 표시
    if len(faces):
        for x, y, w, h in faces:  # 얼굴 좌표와 크기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_4)  # 하얀색 사각형 그리기

    # 4-9. 결과 화면 표시
    cv2.imshow("original", frame)

    # 4-10. 'q' 키 입력 시 루프 종료
    if cv2.waitKey(1000) == ord('q'):
        break

# 5. 종료 처리
capture.release()  # 카메라 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
