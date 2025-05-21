# OpenCV를 사용하여 이미지에서 특정 색상(파란색, 노란색, 초록색)의 네모 상자를 감지하고, 감지된 상자의 중심 좌표를 출력하는 프로그램
# - HSV 색 공간으로 변환하여 각 색상의 범위에 해당하는 마스크를 생성합니다.
# - 마스크에서 윤곽선을 추출하고 바운딩 박스를 계산하여 중심 좌표를 도출합니다.
# 파랑, 노랑, 초록 네모 상자 인식 (+ 좌표/기울어진 각도 출력)
# mask 한 개로 처리한 예

import cv2
import numpy as np

# 1. 특정 색상 네모 상자 감지 함수 정의
def find_color_boxes(image, lower_color, upper_color):
    """
    이미지에서 특정 색상 네모 상자를 감지하고 중심 좌표를 계산.
    :param image: 입력 이미지 (BGR 형식)
    :param lower_color, upper_color: 색상 범위 (HSV 형식)
    :return: 감지된 상자의 중심 좌표 리스트
    """
    # 1-1. 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 1-2. 색상 범위에 대한 마스크 생성
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 1-3. 마스크에서 윤곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1-4. 윤곽선을 기반으로 상자와 중심 좌표 계산
    color_boxes = []
    for contour in contours:
        # 바운딩 박스 계산
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # 넓이 필터링 (노이즈 제거)
        if area > 100:  # 면적 기준
            center_x = x + w // 2  # 중심 X 좌표
            center_y = y + h // 2  # 중심 Y 좌표
            color_boxes.append((center_x, center_y))  # 결과 저장

    return color_boxes

# 2. 이미지 로드
image = cv2.imread('images/box_bgy1.png')  # 입력 이미지 로드

# 3. 색상 범위 설정
# 3-1. 파란색 범위
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# 3-2. 노란색 범위
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 3-3. 초록색 범위
lower_green = np.array([45, 100, 100])
upper_green = np.array([75, 255, 255])

# 4. 특정 색상 네모 상자 감지
blue_boxes = find_color_boxes(image, lower_blue, upper_blue)  # 파란색
yellow_boxes = find_color_boxes(image, lower_yellow, upper_yellow)  # 노란색
green_boxes = find_color_boxes(image, lower_green, upper_green)  # 초록색

# 5. 결과 출력
print("파란색 상자 좌표:", blue_boxes)
print("노란색 상자 좌표:", yellow_boxes)
print("초록색 상자 좌표:", green_boxes)
