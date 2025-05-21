# OpenCV를 사용하여 이미지에서 특정 색상(파란색, 노란색, 초록색)의 네모 상자를 감지하고, 감지된 상자의 중심 좌표를 출력하는 프로그램
# - HSV 색 공간으로 변환하여 각 색상의 범위에 해당하는 마스크를 생성합니다.
# - 마스크에서 윤곽선을 추출하고 바운딩 박스를 계산하여 중심 좌표를 도출합니다.
# 파랑, 노랑, 초록 네모 상자 인식 + 좌표/기울어진 각도 출력
# 굳이 빨강의 경우처럼 mask1, mask2로 작업하여 합친 예 (사실상 불필요함)

############################################################################################################
# 미션 2 :
#    - 이미지에서 파란색, 노란색, 초록색 네모 상자를 감지하세요.
#    - HSV 색상 공간에서 각 색상에 해당하는 범위를 지정하고 마스크를 생성하여 윤곽선을 감지하세요.
#    - 각 윤곽선에 대해 최소 바운딩 박스를 계산하고, 상자의 중심 좌표와 회전 각도를 구하세요.
#    - 감지된 상자 주변에 외곽선을 색상별로 그리세요:
#        - 파란색 상자: 파란 외곽선
#        - 노란색 상자: 노란 외곽선
#        - 초록색 상자: 초록 외곽선
#    - 결과:
#        - 각 색상의 상자 중심 좌표 및 각도를 콘솔에 출력.
#        - 이미지에 각 상자의 외곽선을 그려 시각적으로 결과를 확인.
#    - 조건:
#        - HSV 색상 범위를 정확히 지정하여 노이즈를 최소화.
#        - 윤곽선의 면적 기준(100 픽셀)으로 노이즈 필터링.
#        - 박스는 `cv2.minAreaRect`와 `cv2.boxPoints`를 이용해 그리기.
#    - 주의:
#        - 여러 색상을 한꺼번에 처리하여 코드의 효율성을 유지.
#        - 각 색상별로 구분된 결과를 명확히 표시.
############################################################################################################


# import cv2
# import numpy as np

# # 1. 특정 색상 네모 상자 감지 함수 정의
# def find_color_boxes(image, lower_color1, upper_color1, lower_color2, upper_color2):
#     """
#     이미지에서 특정 색상 네모 상자를 감지하고 중심 좌표를 계산.
#     :param image: 입력 이미지 (BGR 형식)
#     :param lower_color1, upper_color1: 첫 번째 색상 범위 (HSV 형식)
#     :param lower_color2, upper_color2: 두 번째 색상 범위 (HSV 형식)
#     :return: 감지된 상자의 중심 좌표 리스트
#     """
#     # 1-1. 이미지를 HSV 색 공간으로 변환
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # 1-2. 색상 범위에 대한 마스크 생성
#     mask1 = cv2.inRange(hsv, lower_color1, upper_color1)  # 첫 번째 범위
#     mask2 = cv2.inRange(hsv, lower_color2, upper_color2)  # 두 번째 범위
#     mask = cv2.bitwise_or(mask1, mask2)  # 두 마스크 결합

#     # 1-3. 마스크에서 윤곽선 추출
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 1-4. 윤곽선을 기반으로 상자와 중심 좌표 계산
#     color_boxes = []
#     for contour in contours:
#         # 바운딩 박스 계산
#         x, y, w, h = cv2.boundingRect(contour)
#         area = cv2.contourArea(contour)

#         # 넓이 필터링 (노이즈 제거)
#         if area > 100:  # 면적 기준
#             center_x = x + w // 2  # 중심 X 좌표
#             center_y = y + h // 2  # 중심 Y 좌표
#             color_boxes.append((center_x, center_y))  # 결과 저장

#     return color_boxes

# # 2. 이미지 로드
# image = cv2.imread('images/box_bgy1.png')  # 입력 이미지 로드

# # 3. 색상 범위 설정
# # 3-1. 파란색 범위
# lower_blue = np.array([90, 50, 50])
# upper_blue = np.array([130, 255, 255])

# # 3-2. 노란색 범위
# lower_yellow = np.array([20, 100, 100])
# upper_yellow = np.array([30, 255, 255])

# # 3-3. 초록색 범위
# lower_green = np.array([45, 100, 100])
# upper_green = np.array([75, 255, 255])

# # 4. 특정 색상 네모 상자 감지
# blue_boxes = find_color_boxes(image, lower_blue, upper_blue, lower_blue, upper_blue)  # 파란색
# yellow_boxes = find_color_boxes(image, lower_yellow, upper_yellow, lower_yellow, upper_yellow)  # 노란색
# green_boxes = find_color_boxes(image, lower_green, upper_green, lower_green, upper_green)  # 초록색

# # 5. 결과 출력
# print("파란색 상자 좌표:", blue_boxes)
# print("노란색 상자 좌표:", yellow_boxes)
# print("초록색 상자 좌표:", green_boxes)

import cv2
import numpy as np

def find_color_boxes_with_angle(image, lower_color, upper_color, color_name, draw_color):
    """
    이미지에서 특정 색상 네모 상자를 감지하고 중심 좌표와 각도를 계산
    :param image: 입력 이미지 (BGR 형식)
    :param lower_color, upper_color: 색상 범위 (HSV 형식)
    :param color_name: 색상 이름 (출력용)
    :param draw_color: 외곽선 색상 (BGR 형식)
    :return: 감지된 상자의 정보 리스트, 처리된 이미지
    """
    # 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 색상 범위에 대한 마스크 생성
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 마스크에서 윤곽선 추출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 결과 이미지 복사
    result_image = image.copy()
    
    # 윤곽선을 기반으로 상자와 중심 좌표, 각도 계산
    boxes_info = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 넓이 필터링 (노이즈 제거)
        if area > 100:  # 면적 기준
            # 최소 회전 사각형 계산
            rect = cv2.minAreaRect(contour)
            center = rect[0]  # 중심 좌표
            angle = rect[2]  # 각도
            
            # 박스의 4개 꼭지점 계산
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 외곽선 그리기
            cv2.drawContours(result_image, [box], 0, draw_color, 2)
            
            # 중심점 표시
            center_x, center_y = int(center[0]), int(center[1])
            cv2.circle(result_image, (center_x, center_y), 5, draw_color, -1)
            
            # 텍스트 표시 (중심 좌표)
            text = f"{color_name}"
            cv2.putText(result_image, text, 
                       (center_x - 40, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)
            
            # 정보 저장
            box_info = {
                'center': (center_x, center_y),
                'angle': angle,
                'contour': contour
            }
            boxes_info.append(box_info)
    
    return boxes_info, result_image

# 이미지 로드
image = cv2.imread('images/box_bgy1.png')

# 결과 이미지 초기화
result_image = image.copy()

# 색상 설정과 감지
colors = {
    'Blue': {
        'lower': np.array([90, 50, 50]),
        'upper': np.array([130, 255, 255]),
        'draw_color': (255, 0, 0)  # BGR에서 파란색
    },
    'Yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([30, 255, 255]),
        'draw_color': (0, 255, 255)  # BGR에서 노란색
    },
    'Green': {
        'lower': np.array([45, 100, 100]),
        'upper': np.array([75, 255, 255]),
        'draw_color': (0, 255, 0)  # BGR에서 초록색
    }
}

# 결과 출력용 딕셔너리
detection_results = {}

# 각 색상별로 감지 수행
for color_name, color_info in colors.items():
    boxes_info, result_image = find_color_boxes_with_angle(
        result_image,
        color_info['lower'],
        color_info['upper'],
        color_name,
        color_info['draw_color']
    )
    
    detection_results[color_name] = boxes_info
    
    # 결과 출력
    print(f"\n{color_name} 상자 감지 결과:")
    for i, box in enumerate(boxes_info):
        center_x, center_y = box['center']
        angle = box['angle']
        print(f"  {i+1}번째 상자 - 중심 좌표: ({center_x}, {center_y}), 각도: {angle:.2f}°")

# 결과 이미지 표시
cv2.imshow('Color Boxes Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 이미지 저장
cv2.imwrite('result_color_boxes.png', result_image)