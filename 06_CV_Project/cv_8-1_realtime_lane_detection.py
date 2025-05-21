# https://github.com/JD-edu/KG-KAIROS_robot/blob/main/Robot_sensor/python/cv_lane_detect.py
# OpenCV를 활용하여 실시간 카메라 영상을 기반으로 차선 인식을 수행하고, 차선 중심으로부터의 각도를 계산하여 출력하는 프로그램
# - JdOpencvLaneDetect 클래스를 사용하여 차선을 감지하고 차선의 중심으로부터 차량이 기울어진 각도를 계산합니다.
# - 프로그램은 실시간으로 카메라 이미지를 처리하며, 'q' 키를 눌러 종료할 수 있습니다.

# 1. 필요한 모듈 임포트
import cv2
import time
from cv_8_lane_detection_pipeline import JdOpencvLaneDetect  # 차선 감지 클래스

# 2. OpenCV 차선 감지 객체 생성
cv_detector = JdOpencvLaneDetect()  # 차선 감지 클래스 객체 생성

# 3. 카메라 객체 생성 및 해상도 설정
# 3-1. 카메라 객체 생성
cap = cv2.VideoCapture(0)  # 0번 카메라 사용
# 3-2. 카메라 해상도 설정 (320x240)
cap.set(3, 320)  # 가로 해상도 설정
cap.set(4, 240)  # 세로 해상도 설정

# 4. 실시간 차선 감지 및 주행 루틴
while True:
    # 4-1. 약간의 대기 시간
    time.sleep(0.1)

    # 4-2. 카메라에서 이미지 읽기
    ret, img_org = cap.read()
    if ret:  # 이미지가 정상적으로 읽혔는지 확인
        # 4-3. 차선 감지 및 조향 각도 계산
        lanes, img_lane = cv_detector.get_lane(img_org)  # 차선 감지
        angle, img_angle = cv_detector.get_steering_angle(img_lane, lanes)  # 조향 각도 계산

        # 4-4. 차선을 감지하지 못한 경우 처리
        if img_angle is None:
            print("차선을 찾을 수 없습니다...")
            pass
        else:
            # 4-5. 차선 이미지 출력
            cv2.imshow('lane', img_angle)
            print("조향 각도:", angle)

        # 4-6. 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("카메라 오류")

# 5. 프로그램 종료 처리
# 5-1. 카메라 객체 해제
cap.release()
# 5-2. 모든 창 닫기
cv2.destroyAllWindows()
