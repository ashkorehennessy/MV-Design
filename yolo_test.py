import cv2
import time
from ultralytics import YOLO
time_tick = time.time()
time_used = 0.0
# Load the YOLO model
model = YOLO("best_ncnn_model")
# Open the video file
cap = cv2.VideoCapture(2)
# cap.set(3, 320)
# cap.set(4, 240)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        time_used = (time.time() - time_tick) * 0.1 + time_used * 0.9
        time_tick = time.time()
        results = model(frame,imgsz=160)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Inference time: {time_used*1000:.2f}ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Display the annotated frameq
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()