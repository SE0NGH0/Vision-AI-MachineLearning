# OpenCV를 사용하여 실시간으로 웹캠에서 움직임을 감지하고, 움직임이 일정 기준 이상 발생하면 녹화를 시작하는 프로그램

############################################################################################################
# 미션 1:
#    - 웹캠에서 실시간으로 움직임을 감지하여 자동으로 녹화를 시작하세요.
#    - 움직임이 감지되면 최소 5초 동안 동영상을 저장하고, 움직임이 멈추면 녹화를 종료하세요.
#    - 각 프레임에 현재 시간을 표시하여 저장된 영상에 타임스탬프를 추가하세요.
#    - 결과:
#        - 움직임 감지 결과(diff)와 현재 시간을 표시한 프레임을 화면에 출력.
#        - 움직임이 감지되면 "녹화 중..." 메시지 출력.
#        - 녹화된 영상 파일은 'Capture' 폴더에 저장 (예: 녹화_YYYY_MM_DD_HH_MM_SS.avi).
#    - 조건:
#        - 연속된 3개의 프레임을 비교하여 움직임 감지 (absdiff 사용).
#        - 움직임 픽셀 수가 특정 임계값(diff_min)을 초과하면 움직임으로 간주.
#        - 타임스탬프는 영상 상단에 표시 (PIL.ImageFont 사용).
#    - 주의:
#        - 녹화 파일명에 타임스탬프를 포함하여 중복되지 않도록 설정.
############################################################################################################

import cv2
import numpy as np
from datetime import datetime
import os
from PIL import ImageFont, ImageDraw, Image


def add_timestamp(frame, font):
    """
    프레임 상단에 현재 시각을 오버레이합니다.
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((10, 10), timestamp, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    diff_min = 5000  # 움직임 감지 픽셀 임계값
    record_seconds = 5  # 최소 녹화 시간 (초)
    capture_dir = 'Capture'
    os.makedirs(capture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        return

    recording = False
    writer = None
    record_start = None

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    while True:
        ret, frame3 = cap.read()
        if not ret:
            break

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

        diff1 = cv2.absdiff(gray2, gray1)
        diff2 = cv2.absdiff(gray3, gray2)
        motion_mask = cv2.bitwise_and(diff1, diff2)
        _, thresh = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(thresh)

        output_frame = add_timestamp(frame3, font)

        if motion_pixels > diff_min:
            cv2.putText(output_frame, "Recording...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not recording:
                recording = True
                record_start = datetime.now()
                filename = datetime.now().strftime("녹화_%Y_%m_%d_%H_%M_%S.avi")
                filepath = os.path.join(capture_dir, filename)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                height, width = frame3.shape[:2]
                writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        if recording:
            writer.write(output_frame)
            elapsed = (datetime.now() - record_start).total_seconds()
            if elapsed > record_seconds and motion_pixels <= diff_min:
                recording = False
                writer.release()
                writer = None

        cv2.imshow("Diff", motion_mask)
        cv2.imshow("Frame", output_frame)

        frame1, frame2 = frame2, frame3

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
