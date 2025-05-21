
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

import cv2
import numpy as np

# HSV 색상 범위 정의 (lower, upper)
COLOR_RANGES = {
    'blue':  ((100, 150,  50), (140, 255, 255)),
    'yellow':(( 20, 100, 100), ( 30, 255, 255)),
    'green': (( 40,  50,  50), ( 80, 255, 255))
}
# 각 색상별 BGR 그리기 색상
DRAW_COLORS = {
    'blue':  (255,   0,   0),
    'yellow':(  0, 255, 255),
    'green': (  0, 255,   0)
}


def detect_colored_boxes(image_path, min_area=100):
    """
    주어진 이미지에서 파란색, 노란색, 초록색 상자를 검출합니다.
    각 상자의 중심 좌표와 회전 각도를 콘솔에 출력하고,
    이미지에 윤곽선을 그려 화면에 보여줍니다.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    output = img.copy()

    for color, (lower, upper) in COLOR_RANGES.items():
        # 마스크 생성 및 노이즈 제거
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # 최소 바운딩 박스 계산
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            (cx, cy), (_, _), angle = rect

            # 정보 출력
            print(f"{color} 상자: 중심=({cx:.1f}, {cy:.1f}), 각도={angle:.1f}도")

            # 외곽선 그리기
            cv2.drawContours(output, [box], 0, DRAW_COLORS[color], 2)

    # 결과 이미지 표시
    cv2.imshow('Detected Boxes', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = 'images/box_bgy1.png'

    detect_colored_boxes(image_path)
