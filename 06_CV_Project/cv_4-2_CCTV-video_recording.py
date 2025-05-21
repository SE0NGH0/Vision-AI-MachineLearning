## <미션> - 웹캠을 사용하여 실시간 영상에서 녹화 및 캡처를 수행하는 프로그램
# r 키를 눌러 녹화를 시작하거나 중지
# c 키를 눌러 현재 화면을 캡처하여 이미지로 저장
# 녹화 중일 경우 영상에 빨간 점 표시
# q 키를 눌러 프로그램 종료

import cv2
import datetime
from PIL import ImageFont, ImageDraw, Image  # 폰트 처리, 그래픽 요소 추가
import numpy as np

import os
os.makedirs('./result', exist_ok=True)

# 1. 카메라 객체 생성 및 해상도 설정
capt = cv2.VideoCapture(0)  # 기본 웹캠 사용
capt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# 2. 코덱 및 폰트, 녹화 상태 변수 초기화
codex = cv2.VideoWriter_fourcc(*"XVID")  # 코덱 설정 (XVID)
font_ = ImageFont.truetype('fonts/SCDream6.otf', 20)  # 텍스트 표시용 폰트
is_record = False  # 녹화 상태 초기값 (녹화 중 아님)

# 3. 실행 루프 시작
while True:
    # 3-1. 현재 프레임 읽기
    ret, frame = capt.read()

    # 3-2. 현재 시간을 문자열로 저장 (영상에 표시 및 파일명에 사용)
    t_now = datetime.datetime.now()
    t_str = t_now.strftime('%Y/%m/%d %H:%M:%S')  # 화면 표시용 시간
    t_str_path = t_now.strftime('%Y_%m_%d %H_%M_%S')  # 파일명에 사용할 시간

    # 3-3. 화면에 텍스트 배경을 위한 검은 사각형 그리기
    cv2.rectangle(img=frame, pt1=(10, 15), pt2=(340, 35), color=(0, 0, 0), thickness=-1)

    # 3-4. 현재 시간 텍스트를 화면에 추가
    frame = Image.fromarray(frame)  # Numpy 배열 -> Pillow 이미지로 변환
    draw = ImageDraw.Draw(frame)  # Draw 객체 생성
    draw.text(xy=(10, 15), text="보고 있다! " + t_str, font=font_, fill=(255, 255, 255))  # 시간 텍스트 추가
    frame = np.array(frame)  # Pillow 이미지 -> Numpy 배열로 변환

    # 3-5. 키 입력 대기
    key = cv2.waitKey(30)

    # 4. 기능 구현: 키 입력에 따라 동작 수행
    # 4-1. r 키로 녹화 시작 또는 중지
    if key == ord('r') and is_record == False:
        is_record = True  # 녹화 시작 상태로 전환
        # 녹화 파일 생성
        video_ = cv2.VideoWriter(f"./result/Recorded-{t_str_path}.avi", codex, 15, (frame.shape[1], frame.shape[0])) # 녹화 파일 생성

    elif key == ord('r') and is_record == True:
        is_record = False  # 녹화 중지 상태로 전환
        video_.release()  # 녹화 파일 저장 및 종료

    # 4-2. c 키로 현재 화면 캡처
    elif key == ord('c'):
        cv2.imwrite(f"./result/Captured-{t_str_path}.png", frame)  # 캡처된 이미지를 저장

    # 4-3. q 키로 프로그램 종료
    elif key == ord('q'):
        break

    # 5. 녹화 중 표시 (빨간 점 추가)
    if is_record:
        video_.write(frame)  # 현재 프레임을 녹화 파일에 저장
        cv2.circle(img=frame, center=(620, 15), radius=5, color=(0, 0, 255), thickness=-1)  # 빨간 점 표시

    # 6. 영상 출력
    cv2.imshow("Recording in progress", frame)

# 7. 종료 처리
capt.release()  # 카메라 리소스 해제
cv2.destroyAllWindows()  # 모든 창 닫기
