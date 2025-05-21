# ImageProcessor 클래스를 활용하여 웹캠으로부터 실시간 영상을 받아 특정 필터를 선택적으로 적용하는 프로그램 
# cv_3_video_processing_filters_class 파일의 함수들을 가져다 쓰는 예제 파일

import cv2
from cv_3_video_filter_processing_class import ImageProcessor  # ImageProcessor 클래스 임포트

# 1. ImageProcessor 객체 생성
imgEditor = ImageProcessor()  # 새 ImageProcessor 객체 생성 (웹캠 초기화 포함)

# 2. 실행 루프
while True:
    # 2-1. 현재 프레임 읽기
    ret, frame = imgEditor.capt.read()

    # 2-2. 원본 영상 출력
    cv2.imshow("Original2", frame)

    # 2-3. 크기 변경 필터 적용 (2배 확대)
    zoomin = imgEditor.set_size(frame, 2)  # 프레임을 2배로 확대
    cv2.imshow("Zoom In", zoomin)  # 확대된 영상 출력

    # 2-4. 'q' 키 입력 시 루프 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 3. 종료 처리
imgEditor.capt.release()  # 카메라 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
