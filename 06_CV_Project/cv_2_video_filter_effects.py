# OpenCV를 사용하여 실시간 웹캠 영상에 노이즈를 추가하고 다양한 블러 필터(Blur, Gaussian Blur, Bilateral Filter, Median Blur)를 적용하여 그 결과를 비교하는 프로그램
# 노이즈 필터링(salt & pepper에는 medianBlur가 좋다)
import cv2
import numpy as np

# 1. 카메라 객체 생성 및 해상도 설정
capt = cv2.VideoCapture(0)  # 기본 카메라 사용
capt.set(cv2.CAP_PROP_FRAME_WIDTH, 300)  # 가로 해상도 설정
capt.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)  # 세로 해상도 설정

# 2. 실행 루프
while True:
    # 2-1. 현재 프레임 읽기
    ret, frame = capt.read()
    if not ret:
        print("Fail to capture image.")
        break

    # 2-2. 노이즈 생성 및 적용
    noise = np.uint8(np.random.normal(loc=0, scale=0.4, size=frame.shape))  # 정규분포 노이즈 생성
    noised_img = cv2.add(frame, noise)  # 원본 영상에 노이즈 추가

    # 2-3. 다양한 블러 필터 적용
    blur = cv2.blur(noised_img, (5, 5))  # Blur 필터 (5x5 커널)
    gaussian = cv2.GaussianBlur(noised_img, (5, 5), 0)  # Gaussian Blur (표준편차 0)
    bilateral = cv2.bilateralFilter(noised_img, 9, 75, 75)  # Bilateral 필터
    median = cv2.medianBlur(noised_img, 5)  # Median Blur (5x5 필터)

    # 2-4. 개별 영상 출력 (필요에 따라 활성화 가능)
    cv2.imshow("Original", frame)  # 원본 영상
    cv2.imshow("Noised", noised_img)  # 노이즈 추가된 영상
    # cv2.imshow("Blurred", blur)  # Blur 필터 적용 영상
    # cv2.imshow("Gaussian blurred", gaussian)  # Gaussian Blur 적용 영상
    # cv2.imshow("Bilaterally", bilateral)  # Bilateral 필터 적용 영상
    # cv2.imshow("Median Blurred", median)  # Median Blur 적용 영상

    # 2-5. 필터 결과 합쳐서 하나의 창으로 출력
    row1 = cv2.hconcat([frame, noised_img, blur])  # 첫 번째 행: 원본, 노이즈, Blur
    row2 = cv2.hconcat([gaussian, bilateral, median])  # 두 번째 행: Gaussian, Bilateral, Median
    total = cv2.vconcat([row1, row2])  # 두 행을 합침
    cv2.imshow("All in one", total)  # 모든 결과를 하나의 창에 출력

    # 2-6. 종료 조건 확인 ('q' 키 입력 시 종료)
    if cv2.waitKey(1) == ord('q'):
        break

# 3. 종료 처리
capt.release()  # 카메라 객체 해제
cv2.destroyAllWindows()  # 모든 창 닫기
