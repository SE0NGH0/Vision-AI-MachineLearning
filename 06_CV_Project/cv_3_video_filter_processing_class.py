# OpenCV를 활용하여 실시간 웹캠 영상에 대해 색상 필터, 밝기, 대비, 크기 조정을 포함한 여러 필터 효과를 적용하는 프로그램
# 프로그램은 객체지향 방식으로 설계되었으며, ImageProcessor 클래스 내에서 필터와 변환 함수들이 정의되어 있습니다.
# 다양한 필터 효과를 실시간으로 확인할 수 있으며, q 키를 눌러 종료할 수 있습니다.
# 클래스 파일: 외부 파일에서 불러 쓸 수 있게 함수를 모아놓음.

import cv2
import numpy as np

# 1. ImageProcessor 클래스 정의
class ImageProcessor:
    """
    웹캠 영상을 받아와 실시간으로 색상 필터, 밝기 조정, 대비 변경, 크기 조정을 수행하는 클래스.
    """
    def __init__(self, video_source=0, width=640, height=480):
        # 1-1. 카메라 객체 생성 및 해상도 설정
        self.capt = cv2.VideoCapture(video_source)  # 웹캠 연결
        self.capt.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 가로 해상도 설정
        self.capt.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 세로 해상도 설정

    # 2. 색상 변경 함수
    def color_filter(self, img, color, scale):
        """
        특정 색상을 강조하거나 약화시키는 필터.
        img: 입력 이미지, color: 색상 ('red', 'green', 'blue' 또는 0, 1, 2), scale: 강조 비율.
        """
        dst = np.array(img, np.uint8)
        if color == 'blue' or color == 0:
            dst[:, :, 0] = cv2.multiply(dst[:, :, 0], scale)
        elif color == 'green' or color == 1:
            dst[:, :, 1] = cv2.multiply(dst[:, :, 1], scale)
        elif color == 'red' or color == 2:
            dst[:, :, 2] = cv2.multiply(dst[:, :, 2], scale)
        return dst

    # 3. 밝기 변경 함수
    def set_brightness(self, img, scale):
        """
        영상의 밝기를 조정.
        img: 입력 이미지, scale: 밝기 증가값 (양수 = 밝아짐, 음수 = 어두워짐).
        """
        return cv2.add(img, scale)

    # 4. 대비 변경 함수
    def set_contrast(self, img, scale):
        """
        영상의 대비를 조정.
        img: 입력 이미지, scale: 대비 비율 (양수 = 대비 증가, 음수 = 대비 감소).
        """
        return np.uint8(np.clip((1 + scale) * img - 128 * scale, 0, 255))

    # 5. 크기 변경 함수
    def set_size(self, img, scale):
        """
        영상의 크기를 조정.
        img: 입력 이미지, scale: 크기 비율 (2.0 = 2배 확대, 0.5 = 축소).
        """
        return cv2.resize(img, dsize=(int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    # 6. 실시간 필터 적용 루프
    def run_editing(self):
        """
        웹캠 영상에 필터와 변환 효과를 실시간으로 적용.
        """
        while True:
            # 6-1. 현재 프레임 읽기
            ret, frame = self.capt.read()

            # 6-2. 원본 이미지 출력
            cv2.imshow("Original", frame)

            # 6-3. 빨간색 강조 이미지
            redFilter = self.color_filter(frame, 'red', 1.2)
            cv2.imshow("Redder", redFilter)

            # 6-4. 밝기 증가 이미지
            brightened = self.set_brightness(frame, 20)
            cv2.imshow("Brighter", brightened)

            # 6-5. 대비 변경 이미지
            contrast = self.set_contrast(frame, 0.9)
            cv2.imshow("Contrast", contrast)

            # 6-6. 크기 변경 이미지
            zoomin = self.set_size(frame, 2)
            cv2.imshow("Bigger", zoomin)

            # 6-7. 'q' 키 입력 시 루프 종료
            if cv2.waitKey(1) == ord('q'):
                break

        # 7. 종료 처리
        self.capt.release()  # 카메라 객체 해제
        cv2.destroyAllWindows()  # 모든 창 닫기

# 8. 프로그램 실행
if __name__ == "__main__":
    """
    프로그램 실행 엔트리 포인트.
    ImageProcessor 클래스의 객체를 생성하고 필터 적용 루프를 실행.
    """
    processor = ImageProcessor()  # ImageProcessor 객체 생성
    processor.run_editing()  # 실시간 필터 적용 루프 실행
