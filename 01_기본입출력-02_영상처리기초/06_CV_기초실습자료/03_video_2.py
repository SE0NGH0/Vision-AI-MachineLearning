# https://opencv-python.readthedocs.io/en/latest/doc/02.videoStart/videoStart.html 
# 화면 크기 조절

import cv2

vCap = cv2.VideoCapture(0)

print('width: {0}, height: {1}'.format(vCap.get(3), vCap.get(4)))    # 3은 디폴트 width: 640.0, 4는 height: 480.0

vCap.set(3, 920) # 화면 가로(3) 크기 조절
vCap.set(4, 540) # 화면 세로(4) 크기 조절

while(True):
    ret, frame = vCap.read() #ret: 캡쳐 결과(T/F), frame: 캡쳐한 프레임

    if(ret):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capture", img_gray) 

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

vCap.release()
cv2.destroyAllWindows()
