# File로 영상 재생. 캠 촬영과 동일함.

import cv2

vCap = cv2.VideoCapture('images/capture.avi')

while vCap.isOpened():
    ret, frame = vCap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Capture', frame)
    
    # 초당 30프레임의 경우 적정 waitKey 인자값
    # 대기 시간 (밀리초) = (1000 / 30) ≈ 33
    if cv2.waitKey(33) & 0xff == ord('q'):
        break
vCap.release()
cv2.destroyAllWindows()
