import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n_ncnn_model")
ncnn_model_imgsz = 128
# Open the video file
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, imgsz=128, int8=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

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