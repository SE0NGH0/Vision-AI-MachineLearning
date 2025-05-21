# https://opencv-python.readthedocs.io/en/latest/doc/02.videoStart/videoStart.html 

import cv2

vCap = cv2.VideoCapture(0) # 0번 디폴트 카메라

while True:
    ret, frame = vCap.read() # ret: 캡쳐결과(T/F), frame: 캡쳐한 프레임

    if ret:
        cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vCap.release()
cv2.destroyAllWindows()
