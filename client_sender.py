# insta360 camera에서 받은 데이터를 WebRTC로 서버에 전송하는 프로그램

import asyncio
#import websockets
import json
import time
import threading
import socket
import sys
import os
import cv2
import numpy as np
import base64
import time
import datetime
import cv2

# insta360 camera capture using opencv video capture
cam = cv2.VideoCapture(cv2.CAP_ANY) # 0번 카메라 장치 연결

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # 앞, 뒤로 나뉘어진 360 video를 합쳐서 equirectangular로 변환ㅇ    
    
    cv2.imshow('camera', frame)
    if cv2.waitKey(25) != -1: # q를 누르면 종료
        break
    
cam.release()
cv2.destroyAllWindows()
