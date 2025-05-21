# OpenCV를 사용하여 웹캠에서 실시간 영상을 처리하고, 영상의 색상, 밝기, 대비, 크기 등을 변경하는 다양한 필터와 변환 기능을 구현하는 프로그램

import cv2
import numpy as np

# 1. 색상 변경 함수 설정
def color_filter(img, color, scale):  # (입력 영상, 색상, 비율)
    """
    이미지의 특정 색상을 강조하거나 약화시킴.
    color: 'blue', 'green', 'red' 또는 0, 1, 2 (B, G, R 채널 인덱스)
    scale: 강조 비율 (1.0 = 원본, 1.2 = 강조, <1 = 약화)
    """
    dst = np.array(img, np.uint8)  # 입력 영상을 복제
    if color == 'blue' or color == 0:  # 파란색 강조
        dst[:, :, 0] = cv2.multiply(dst[:, :, 0], scale)
    elif color == 'green' or color == 1:  # 초록색 강조
        dst[:, :, 1] = cv2.multiply(dst[:, :, 1], scale)
    elif color == 'red' or color == 2:  # 빨간색 강조
        dst[:, :, 2] = cv2.multiply(dst[:, :, 2], scale)
    return dst  # 변경된 영상 반환

# 2. 밝기 변경 함수 설정
def set_brightness(img, scale):
    """
    영상의 밝기를 조정.
    scale: 밝기 증가 값 (양수 = 밝아짐, 음수 = 어두워짐)
    """
    return cv2.add(img, scale)

# 3. 대비 변경 함수 설정
def set_contrast(img, scale):
    """
    영상의 대비를 조정.
    scale: 대비 비율 (양수 = 대비 증가, 음수 = 대비 감소)
    """
    return np.uint8(np.clip((1 + scale) * img - 128 * scale, 0, 255))

# 4. 이미지 크기 변경 함수 설정
def set_size(img, scale):
    """
    영상의 크기를 조정.
    scale: 크기 비율 (2.0 = 2배 확대, 0.5 = 50% 축소)
    """
    return cv2.resize(img, dsize=(int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

# 5. 카메라 객체 생성 및 해상도 설정
capt = cv2.VideoCapture(0)  # 0번 카메라
capt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# 6. 실행 루프
while True:
    # 6-1. 현재 프레임 읽기
    ret, frame = capt.read()  # ret: 성공 여부, frame: 현재 프레임

    # 6-2. 원본 이미지 출력
    cv2.imshow("original", frame)

    # 6-3. 특정 필터 적용 (필요 시 주석 해제하여 활성화)
    
    # 빨간색 강조 (강조 비율: 1.2)
    # redFilter = color_filter(frame, 'red', 1.2)
    # cv2.imshow("Redder", redFilter)

    # 밝기 증가 (밝기 값: 20)
    # brightened = set_brightness(frame, 20)
    # cv2.imshow("Brighter", brightened)

    # 대비 증가 (대비 값: 0.9)
    # contrast = set_contrast(frame, 0.9)
    # cv2.imshow("Contrast", contrast)

    # 크기 변경 (2배 확대)
    zoomin = set_size(frame, 2)
    cv2.imshow("Bigger", zoomin)

    # 6-4. 키 입력 대기 및 종료 조건 확인
    if cv2.waitKey(1) == ord('q'):  # 1ms 대기 후 'q' 키 확인
        break

# 7. 종료 처리
capt.release()  # 카메라 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
