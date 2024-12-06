import cv2
import time
time_used = 0.0
width = 320
height = 240
video = cv2.VideoCapture('/dev/video0')
video.set(3, width)
video.set(4, height)
while True:
    timestamp = time.time()
    ret, image = video.read()
    time_used = (time.time() - timestamp) * 0.1 + time_used * 0.9
    cv2.putText(image, f"capture time: {time_used:.4f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("video", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
