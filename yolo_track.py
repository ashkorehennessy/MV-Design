from collections import defaultdict

import cv2
import time

import numpy as np
from ultralytics import YOLO
time_tick = time.time()
time_used = 0.0
# Load the YOLO model
model = YOLO("best_ncnn_model")
# Open the video file
cap = cv2.VideoCapture(2)
# cap.set(3, 320)
# cap.set(4, 240)
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        time_used = (time.time() - time_tick) * 0.1 + time_used * 0.9
        time_tick = time.time()
        results = model.track(frame,imgsz=160,persist=True)

        annotated_frame = results[0].plot()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Visualize the results on the frame
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=5)

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