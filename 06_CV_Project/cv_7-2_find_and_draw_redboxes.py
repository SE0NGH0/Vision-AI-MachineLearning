# OpenCV를 사용하여 이미지에서 빨간색 네모 상자를 감지하고, 감지된 상자를 이미지에 그린 후 중심 좌표와 회전 각도를 출력하는 프로그램
# - 이미지를 HSV 색 공간으로 변환하여 빨간색 범위를 마스킹합니다.
# - 윤곽선을 추출하여 최소 바운딩 박스를 계산하고, 이를 이미지에 그립니다.
# - 중심 좌표와 각도를 계산하여 출력합니다.
# 기존 + 창 띄워서 보기

import cv2
import numpy as np

# 1. 빨간 네모 상자 감지 함수 정의
def find_red_boxes(image):
    """
    입력 이미지에서 빨간색 네모 상자를 감지하고 중심 좌표와 각도를 계산.
    :param image: 입력 이미지 (BGR 형식)
    :return: 빨간 네모 상자의 중심 좌표와 각도 리스트, 박스가 그려진 이미지
    """
    # 1-1. 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1-2. 빨간색 범위 지정
    lower_red = np.array([0, 100, 100])    # 빨간색 하한
    upper_red = np.array([10, 255, 255])   # 빨간색 상한
    lower_red2 = np.array([170, 100, 100]) # 빨간색 하한 (반대쪽 끝)
    upper_red2 = np.array([180, 255, 255]) # 빨간색 상한

    # 1-3. 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)  # 두 마스크 결합

    # 1-4. 마스크에서 윤곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1-5. 윤곽선을 기반으로 빨간 네모 상자와 각도 계산
    red_boxes = []
    for contour in contours:
        # 바운딩 박스 계산
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # 네 꼭짓점 좌표
        box = np.int0(box)         # 정수형 변환

        # 바운딩 박스 그리기
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # 면적 기준 필터링 (노이즈 제거)
        if cv2.contourArea(contour) > 100:
            # 중심 좌표 계산
            M = cv2.moments(contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            angle = rect[2]
            red_boxes.append((center_x, center_y, angle))

    return red_boxes, image

# 2. 이미지 로드
image = cv2.imread('images/box_r1.png')  # 입력 이미지 로드

# 3. 빨간 네모 상자 감지
red_boxes, image_with_boxes = find_red_boxes(image)

# 4. 결과 출력
for box in red_boxes:
    print("빨간 네모 상자의 좌표:", (box[0], box[1]))
    print("빨간 네모 상자의 각도:", box[2])

# 5. 결과 이미지 출력
cv2.imshow('Image with Red Boxes', image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
