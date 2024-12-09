import cv2
import time
import multiprocessing
from ultralytics import YOLO

def capture_frames(frame_queue):
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if not frame_queue.full():
                frame_queue.put(frame)
        else:
            break
    cap.release()

def yolo_inference(frame_queue):
    model = YOLO("yolo11n_ncnn_model")
    time_tick = time.time()
    time_used = 0.0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            time_used = (time.time() - time_tick) * 0.1 + time_used * 0.9
            time_tick = time.time()
            results = model(frame,imgsz=128)
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"Inference time: {time_used*1000:.2f}ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = multiprocessing.Queue(maxsize=10)
    capture_process = multiprocessing.Process(target=capture_frames, args=(frame_queue,))
    inference_process = multiprocessing.Process(target=yolo_inference, args=(frame_queue,))

    capture_process.start()
    inference_process.start()

    capture_process.join()
    inference_process.join()