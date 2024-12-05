import cv2
import time
import subprocess
import numpy as np
import multiprocessing
import signal

width = 320
height = 240
video = cv2.VideoCapture('/dev/video0')
video.set(3, width)
video.set(4, height)
while True:
    ret, image = video.read()
    cv2.imshow("video", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
