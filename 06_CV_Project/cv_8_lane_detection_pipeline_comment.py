import cv2  # OpenCV 라이브러리 불러오기 (컴퓨터 비전 기능 사용)
import numpy as np  # 수치 계산을 위한 NumPy 불러오기
import logging  # 로그 출력을 위한 모듈
import math  # 수학 함수 사용을 위한 모듈
import datetime  # (현재는 사용되지 않으나) 시간 관련 기능 불러오기
import sys  # (현재는 사용되지 않으나) 시스템 관련 기능 불러오기

show_image = False  # 이미지 출력 여부를 결정하는 전역 플래그

class JdOpencvLaneDetect(object):  # 차선 감지 및 조향 계산을 담당하는 클래스 정의
    def __init__(self):  # 생성자
        self.curr_steering_angle = 90  # 초기 조향각을 90도로 설정 (직진)

    def get_lane(self, frame):  # 프레임에서 차선 검출 수행
        show_image("original", frame)  # 원본 프레임 표시 (옵션)
        lane_lines, frame = detect_lane(frame)  # 차선 좌표와 차선 그린 이미지를 반환
        return lane_lines, frame  # 차선 좌표, 시각화된 프레임 반환

    def get_steering_angle(self, img_lane, lane_lines):  # 시각화된 이미지와 차선 좌표로 조향각 계산
        if len(lane_lines) == 0:  # 차선을 못 찾았으면
            return 0, None  # 각도 0, 이미지 없음 반환
        # 새 조향각 계산
        new_steering_angle = compute_steering_angle(img_lane, lane_lines)
        # 조향각 안정화 (급격한 변화 제한)
        self.curr_steering_angle = stabilize_steering_angle(
            self.curr_steering_angle,
            new_steering_angle,
            len(lane_lines)
        )

        curr_heading_image = display_heading_line(img_lane, self.curr_steering_angle)  # 조향 방향선 시각화
        show_image("heading", curr_heading_image)  # 시각화된 조향선 이미지 표시

        return self.curr_steering_angle, curr_heading_image  # 최종 각도, 시각화 이미지 반환

############################
# Frame processing steps
############################

def detect_lane(frame):  # 전체 차선 검출 파이프라인
    logging.debug('detecting lane lines...')  # 디버그 로그 출력
    edges = detect_edges(frame)  # 엣지(경계) 검출
    show_image('edges', edges)  # 검출된 엣지 표시

    cropped_edges = region_of_interest(edges)  # 관심 영역(하단 절반)만 남김
    show_image('edges cropped', cropped_edges, True)  # 잘린 영역 표시

    line_segments = detect_line_segments(cropped_edges)  # 허프 변환으로 선분 검출
    line_segment_image = display_lines(frame, line_segments)  # 선분 시각화
    show_image("line segments", line_segment_image)  # 선분 표시

    lane_lines = average_slope_intercept(frame, line_segments)  # 선분을 기반으로 차선 평균화
    lane_lines_image = display_lines(frame, lane_lines)  # 평균화된 차선 시각화
    show_image("lane lines images", lane_lines_image)  # 차선 표시
  
    return lane_lines, lane_lines_image  # 차선 좌표, 시각화된 이미지 반환

''' To improve red line detection
1. change hue value: lower_red1[0], upper_red1[0], lower_red2[0], upper_red2[0]
   recommand values are 170 ~ 180 and 0 ~ 30. we use 2 masks.
2. change saturation value: lower_red1[1], lower_red2[1]
   recommand values: 70 ~ 100
3. change value value: lower_red1[2], lower_red2[2]
   recommand values: 30 ~ 100
'''

def detect_edges(frame):  # 엣지 검출 함수
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR 색공간을 HSV로 변환
    show_image("hsv", hsv)  # HSV 이미지 표시
    # 빨간색 영역 마스크 1 (저구간)
    lower_red1 = np.array([0, 50, 50])  # H=0, S=50, V=50
    upper_red1 = np.array([40, 255, 255])  # H=40까지
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # 첫 번째 마스크
    # 빨간색 영역 마스크 2 (고구간)
    lower_red2 = np.array([160, 50, 50])  # H=160
    upper_red2 = np.array([180, 255, 255])  # H=180까지
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # 두 번째 마스크
    mask = mask1 + mask2  # 두 마스크 합치기

    show_image("red mask", mask, True)  # 빨간색 마스크 표시

    edges = cv2.Canny(mask, 200, 400)  # Canny 엣지 검출 (임계치 200,400)
    show_image("red edge", edges)  # 엣지 이미지 표시

    return edges  # 엣지 이미지 반환


def region_of_interest(canny):  # 관심 영역 마스크 함수
    height, width = canny.shape  # 이미지 크기 추출
    mask = np.zeros_like(canny)  # 동일 크기의 0 행렬 생성

    # 화면 하단 절반을 폴리곤으로 정의
    polygon = np.array([[  
        (0, height*(1/2)),  # 왼쪽 중간
        (width, height*(1/2)),  # 오른쪽 중간
        (width, height),  # 오른쪽 하단
        (0, height),  # 왼쪽 하단
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)  # 폴리곤 영역을 흰색(255)으로 채우기
    show_image("mask", mask)  # 마스크 표시
    masked_image = cv2.bitwise_and(canny, mask)  # 관심 영역 이외는 제거
    return masked_image  # 관심 영역 엣지 반환


def detect_line_segments(cropped_edges):  # 허프 변환으로 선분 검출
    rho = 1  # 거리 정밀도: 1px
    angle = np.pi / 180  # 각도 정밀도: 1도
    min_threshold = 10  # 최소 투표 수
    # 선분 검출: 최소 길이 15, 최대 간격 4
    line_segments = cv2.HoughLinesP(
        cropped_edges, rho, angle, min_threshold,
        np.array([]), minLineLength=15, maxLineGap=4
    )

    if line_segments is not None:  # 선분이 검출되었으면
        for line_segment in line_segments:
            logging.debug('detected line_segment:')  # 디버그 로그
            logging.debug(
                "%s of length %s" % (
                    line_segment,
                    length_of_line_segment(line_segment[0])
                )
            )  # 선분 좌표와 길이 출력

    return line_segments  # 선분 배열 반환


def average_slope_intercept(frame, line_segments):  # 선분 평균화로 차선 하나/두 개 생성
    lane_lines = []  # 최종 차선 리스트
    if line_segments is None:  # 검출된 선분 없으면
        logging.info('No line_segment segments detected')  # 정보 로그
        return lane_lines  # 빈 리스트 반환

    height, width, _ = frame.shape  # 원본 프레임 크기
    left_fit = []  # 좌측 기울기/절편 저장
    right_fit = []  # 우측 기울기/절편 저장

    boundary = 1/3  # 화면 1/3 경계
    left_region_boundary = width * (1 - boundary)  # 왼쪽 2/3 경계값
    right_region_boundary = width * boundary  # 오른쪽 1/3 경계값
    
    for line_segment in line_segments:  # 모든 선분 반복
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:  # 수직선 무시
                logging.info(
                    'skipping vertical line segment (slope=inf): %s' % line_segment
                )
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 기울기(slope), 절편(intercept) 계산
            slope, intercept = fit  # 결과 언패킹
            if slope < 0:  # 기울기가 음수면 좌측 후보
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    if slope < -0.75:  # 너무 완만한 선 제외
                        left_fit.append((slope, intercept))  # 좌측 리스트에 추가
            else:  # 기울기가 양수면 우측 후보
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    if slope > 0.75:  # 너무 완만한 선 제외
                        right_fit.append((slope, intercept))  # 우측 리스트에 추가

    # 좌측 평균 기울기/절편
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)  # 평균 계산
        lane_lines.append(make_points(frame, left_fit_average))  # 좌측 차선 좌표 생성

    # 우측 평균 기울기/절편
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)  # 평균 계산
        lane_lines.append(make_points(frame, right_fit_average))  # 우측 차선 좌표 생성

    logging.debug('lane lines: %s' % lane_lines)  # 최종 차선 좌표 로그

    return lane_lines  # 차선 리스트 반환

 
def compute_steering_angle(frame, lane_lines):  # 조향각 계산 함수
    if len(lane_lines) == 0:  # 차선 없으면
        logging.info('No lane lines detected, do nothing')
        return -90  # 기본값 -90° 반환

    height, width, _ = frame.shape  # 프레임 크기
    if len(lane_lines) == 1:  # 한 개 차선만 검출 시
        x1, _, x2, _ = lane_lines[0][0]  # 차선 선분 좌표 추출
        x_offset = x2 - x1  # x 이동량
    else:  # 두 개 차선 검출 시
        _, _, left_x2, _ = lane_lines[0][0]  # 좌측 선 끝점
        _, _, right_x2, _ = lane_lines[1][0]  # 우측 선 끝점
        # 카메라 중앙 보정 비율
        camera_mid_offset_percent = 0.02
        mid = int(width / 2 * (1 + camera_mid_offset_percent))  # 보정된 화면 중앙
        x_offset = (left_x2 + right_x2) / 2 - mid  # 중앙선 이동량

    y_offset = int(height / 2)  # y축 이동량: 화면 중간 높이

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # 라디안 단위 각도
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # 도 단위 변환
    steering_angle = angle_to_mid_deg + 90  # 90° 기준으로 변환

    logging.debug('new steering angle: %s' % steering_angle)  # 로그
    return steering_angle  # 계산된 조향각 반환


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    """
    이전 각도를 이용해 새 각도 변화를 제한하여 급격한 회전을 방지
    """
    # 차선 수에 따라 허용 오차 설정
    if num_of_lane_lines == 2 :
        max_angle_deviation = max_angle_deviation_two_lines  # 두 선 검출 시
    else :
        max_angle_deviation = max_angle_deviation_one_lane  # 한 선 검출 시
    
    angle_deviation = new_steering_angle - curr_steering_angle  # 변화량
    if abs(angle_deviation) > max_angle_deviation:
        # 최대 오차만큼만 이동
        stabilized_steering_angle = int(
            curr_steering_angle + max_angle_deviation * angle_deviation / abs(angle_deviation)
        )
    else:
        stabilized_steering_angle = new_steering_angle  # 변화량이 작으면 그대로 사용
    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle  # 안정화된 각도 반환


"""
  Utility Functions
"""

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):  # 선 시각화
    line_image = np.zeros_like(frame)  # 빈 이미지 생성
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)  # 선 그리기
    # 원본과 가중치 합성
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image  # 합성 이미지 반환


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):  # 조향선 시각화
    heading_image = np.zeros_like(frame)  # 빈 이미지
    height, width, _ = frame.shape  # 크기

    steering_angle_radian = steering_angle / 180.0 * math.pi  # 라디안 변환
    x1 = int(width / 2)  # 시작점 x: 화면 중앙
    y1 = height  # 시작점 y: 화면 하단
    # 삼각함수로 끝점 계산
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)  # 끝점 y: 화면 중간

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)  # 조향선 그리기
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)  # 합성

    return heading_image  # 조향선 시각화 이미지 반환


def length_of_line_segment(line):  # 선분 길이 계산
    x1, y1, x2, y2 = line  # 좌표 언패킹
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 피타고라스


def show_image(title, frame, show=show_image):  # 이미지 표시 함수
    if show:  # 전역 플래그가 True일 때
        cv2.imshow(title, frame)  # 창에 이미지 출력


def make_points(frame, line):  # (기울기, 절편) → 실제 좌표 변환
    height, width, _ = frame.shape  # 크기 추출
    slope, intercept = line  # 기울기, 절편 언패킹
    y1 = height  # 프레임 하단 y 좌표
    y2 = int(y1 * 1 / 2)  # 중간 높이 y 좌표

    # x 좌표 계산 및 프레임 범위 클램핑
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]  # 좌표 리스트 반환
