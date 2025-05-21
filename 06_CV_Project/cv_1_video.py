# 가상환경 구축: python -m venv myenv
# 가상환경 활성화: myenv\Scripts\activate.bat
# 가상환경 비활성화: myenv\Scripts\deactivate.bat

# OpenCV-python 설치: pip install opencv-python
# python -m pip install --upgrade pip  

# haarcascade .xml 확인: myenv\Lib\cv2\data  

import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("Original", frame)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
