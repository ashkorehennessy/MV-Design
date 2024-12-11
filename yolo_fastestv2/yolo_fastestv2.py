import subprocess
import struct
import threading
import json
import cv2
import select

class YoloFastestV2:
    def __init__(self, exe_path, timeout=5.0):
        self.timeout = timeout  # 超时时间（秒）
        self.proc = subprocess.Popen(
            exe_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        self.stdout_thread = threading.Thread(target=self._consume_stderr, daemon=True)
        self.stdout_thread.start()

    def _consume_stderr(self):
        while True:
            line = self.proc.stderr.readline()
            if not line:
                break
            print("C++ Stderr:", line.decode().strip())  # 调试错误

    def infer(self, frame):
        success, encoded_image = cv2.imencode(".jpg", frame)
        if not success:
            print("Error: Failed to encode image")
            return None

        img_bytes = encoded_image.tobytes()
        img_size = len(img_bytes)

        # 发送图像数据到子进程
        self.proc.stdin.write(struct.pack("I", img_size))
        self.proc.stdin.write(img_bytes)
        self.proc.stdin.flush()  # 确保写入完成

        # 等待子进程返回结果
        result = self._read_with_timeout()
        if result is None:
            print("Error: Subprocess did not respond in time")
            return None

        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            print("Error decoding JSON result:", result)
            result = None

        return result

    def _read_with_timeout(self):
        poll = select.poll()
        poll.register(self.proc.stdout, select.POLLIN)
        events = poll.poll(self.timeout * 1000)  # 超时时间以毫秒为单位

        if events:
            line = self.proc.stdout.readline().strip()
            return line.decode()
        else:
            # 超时未返回
            return None

    def close(self):
        self.proc.stdin.write(struct.pack("I", 0))
        self.proc.stdin.flush()
        self.proc.stdin.close()
        self.proc.terminate()

# 示例使用
if __name__ == "__main__":
    yolo_proc = YoloFastestV2("./yolo_fastestv2_aarch64")
    cap = cv2.VideoCapture(0)  # 打开摄像头

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = yolo_proc.infer(frame)
        print("Inference result:", result)

        for box in result or []:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = f'{box["class"]}: {box["score"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO FastestV2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    yolo_proc.close()
    cap.release()
    cv2.destroyAllWindows()
