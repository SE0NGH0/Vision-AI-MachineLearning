# 웹캠으로 실시간 영상을 받아 현재 시간 정보를 영상 위에 표시하며, 사용자가 q 키를 누를 때까지 영상을 화면에 출력하는 프로그램
# 준비물: fonts 폴더 만들어서 폰트 파일 넣어두기


## 방법1 : Pillow (Draw 객체의 text() 메서드사용 ##########################################################################

# import cv2
# import datetime
# from PIL import ImageFont, ImageDraw, Image  # 폰트 처리, 텍스트 추가, Pillow 사용
# import numpy as np

# # 1. VideoCapture 객체 생성 및 해상도 설정
# capt = cv2.VideoCapture(0)  # 0번 카메라
# capt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
# capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# # 2. 폰트 로드
# font = ImageFont.truetype("fonts/SCDream6.otf", 20)  # 텍스트 표시를 위한 폰트 로드

# # 3. 실행 루프
# while True:
#     # 3-1. 프레임 읽기
#     ret, frame = capt.read()

#     # 3-2. 현재 시간 가져오기 및 문자열로 변환
#     t_now = datetime.datetime.now()
#     t_now_show = t_now.strftime('%Y/%m/%d %H:%M:%S')  # 텍스트 형식의 현재 시각

#     # 3-3. 텍스트 배경용 사각형 추가
#     cv2.rectangle(img=frame, pt1=(10, 15), pt2=(350, 35), color=(0, 0, 0), thickness=-1)
#     # img: 대상 이미지, pt1, pt2: 사각형 시작/끝 점, (b, g, r): 색상, thickness: 두께 (-1: 채우기)

#     # 3-4. 영상에 텍스트 추가 (Pillow 사용)
#     frame = Image.fromarray(frame)  # Numpy 배열을 Pillow 이미지로 변환
#     draw = ImageDraw.Draw(frame)  # Draw 객체 생성
#     draw.text(xy=(10, 15), text="지켜보고 있다! " + t_now_show, font=font, fill=(255, 255, 255))
#     # xy: 시작 위치, text: 표시할 텍스트, font: 폰트, fill: 글자 색상
#     frame = np.array(frame)  # Pillow 이미지를 다시 Numpy 배열로 변환

#     # 3-5. 프레임 화면에 출력
#     cv2.imshow("CCTV", frame)

#     # 3-6. 'q' 키 입력 시 루프 종료
#     if cv2.waitKey(1) == ord('q'):  # 1ms 대기 후 키 입력 확인
#         break

# # 4. 종료 처리
# capt.release()  # 카메라 리소스 해제
# cv2.destroyAllWindows()  # 모든 창 닫기


## 방법2 : Opencv (cv2.putText() 메서드 기능 사용 ############################################################################
import cv2
import datetime
import numpy as np

# 1. VideoCapture 객체 생성 및 해상도 설정
capt = cv2.VideoCapture(0)  # 0번 카메라
capt.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정

# 2. 폰트 설정
font = cv2.FONT_HERSHEY_SIMPLEX  # OpenCV 내장 폰트 사용
font_scale = 0.7  # 폰트 크기 설정
font_color = (255, 255, 255)  # 흰색 글자
font_thickness = 2  # 글자 두께
text_position = (10, 30)  # 텍스트 시작 위치 (x, y)

# 3. 실행 루프
while True:
    # 3-1. 프레임 읽기
    ret, frame = capt.read()

    # 3-2. 현재 시간 가져오기 및 문자열로 변환
    t_now = datetime.datetime.now()
    t_now_show = t_now.strftime('%Y/%m/%d %H:%M:%S')  # 텍스트 형식의 현재 시각

    # 3-3. 텍스트 배경용 사각형 추가
    cv2.rectangle(img=frame, pt1=(5, 5), pt2=(500, 40), color=(0, 0, 0), thickness=-1)
    # img: 대상 이미지, pt1, pt2: 사각형 시작/끝 점, (b, g, r): 색상, thickness: 두께 (-1: 채우기)

    # 3-4. 영상에 텍스트 추가 (OpenCV 사용)
    cv2.putText(img=frame, text="I'm watching you!  " + t_now_show, org=text_position, fontFace=font,
                fontScale=font_scale, color=font_color, thickness=font_thickness, lineType=cv2.LINE_AA)
    # img: 대상 이미지, text: 표시할 텍스트, org: 텍스트 시작 위치, fontFace: 폰트, fontScale: 크기, color: 색상, thickness: 두께

    # 3-5. 프레임 화면에 출력
    cv2.imshow("CCTV", frame)

    # 3-6. 'q' 키 입력 시 루프 종료
    if cv2.waitKey(1) == ord('q'):  # 1ms 대기 후 키 입력 확인
        break

# 4. 종료 처리
capt.release()  # 카메라 리소스 해제
cv2.destroyAllWindows()  # 모든 창 닫기

