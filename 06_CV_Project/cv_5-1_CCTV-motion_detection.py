# CCTV: 웹캠에서 실시간으로 프레임을 읽고, 연속된 세 프레임 간의 차이를 계산하여 움직임을 감지하는 프로그램
import cv2
import datetime
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# 1. 프레임 비교 함수 정의
def get_diff_img(frame_a, frame_b, frame_c, threshold):
    """
    세 개의 프레임을 비교하여 움직임 감지.
    frame_a, frame_b, frame_c: 비교할 연속된 세 프레임
    threshold: 차이값을 흑백 변환 기준
    """
    # 1-1. 프레임을 그레이스케일로 변환
    frame_a_gray = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    frame_b_gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    frame_c_gray = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)

    # 1-2. 연속된 프레임 간 차이 계산
    diff_ab = cv2.absdiff(frame_a_gray, frame_b_gray)
    diff_bc = cv2.absdiff(frame_b_gray, frame_c_gray)

    # 1-3. 차이를 이진화 (threshold 이상인 부분만 감지)
    _, diff_ab_t = cv2.threshold(diff_ab, threshold, 255, cv2.THRESH_BINARY)
    _, diff_bc_t = cv2.threshold(diff_bc, threshold, 255, cv2.THRESH_BINARY)

    # 1-4. 두 차이를 AND 연산으로 결합
    diff = cv2.bitwise_and(diff_ab_t, diff_bc_t)

    # 1-5. 노이즈 제거 (MORPH_OPEN)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

    # 1-6. 움직임이 감지된 픽셀 수 계산
    diff_cnt = cv2.countNonZero(diff)
    return diff, diff_cnt


# 2. 초기 설정
capt = cv2.VideoCapture(0)  # 카메라 객체 생성
capt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도 설정 (가로)
capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 해상도 설정 (세로)

# 2-1. 폰트 및 파일 설정
font = ImageFont.truetype('fonts/SCDream6.otf', 20)  # 텍스트 표시용 폰트
threshold = 40  # 움직임 감지 임계값
diff_min = 10  # 움직임 픽셀 최소 수

# 3. 초기 비교 프레임 설정
ret, frame_a = capt.read()  # 첫 번째 프레임
ret, frame_b = capt.read()  # 두 번째 프레임 (1프레임 이후)

# 4. 실행 루프
while True:
    # 4-1. 현재 프레임 읽기
    ret, frame_c = capt.read()
    frame = np.array(frame_c)

    # 4-2. 현재 시각 표시를 위한 텍스트 생성
    t_now = datetime.datetime.now()
    t_str = t_now.strftime('%Y/%m/%d %H:%M:%S')
    t_str_path = t_now.strftime('%Y_%m_%d %H_%M_%S')

    # 4-3. 프레임 상단에 검은 배경 및 텍스트
    cv2.rectangle(img=frame, pt1=(10, 15), pt2=(340, 35), color=(0, 0, 0), thickness=-1)

    # 4-4. 움직임 감지
    diff, diff_cnt = get_diff_img(frame_a=frame_a, frame_b=frame_b, frame_c=frame_c, threshold=threshold)

    # 4-5. 움직임이 일정 기준 이상인 경우 캡처 저장
    if diff_cnt > diff_min:
        cv2.imwrite("Capture/captured" + t_str_path + ".png", frame)

    # 4-6. 움직임 감지 결과 영상 디스플레이 
    cv2.imshow("diff", diff)

    # 4-7. 텍스트를 프레임에 삽입
    frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    draw.text(xy=(10, 15), text="보고있다~ " + t_str, font=font, fill=(255, 255, 255))
    frame = np.array(frame)

    # 4-8. 프레임 업데이트 (이전 프레임을 현재 프레임으로)
    frame_a = np.array(frame_b)
    frame_b = np.array(frame_c)

    # 4-9. 키 입력 대기 (30ms)
    key = cv2.waitKey(30)
    if key == ord('q'):  # 'q' 키를 누르면 종료
        break

    # 4-10. 현재 프레임 디스플레이
    cv2.imshow("Original", frame)

# 5. 종료 처리
capt.release()  # 카메라 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
