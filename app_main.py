import sys
from collections import defaultdict

import cv2
import time
import glob
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, \
    QMessageBox, QComboBox, QCheckBox, QTextEdit
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from yolo_fastestv2.yolo_fastestv2 import YoloFastestV2
import platform

width = 320
height = 240
export_imgsz = 128

# Simple user database
USER_DATABASE = {
    "admin": "password123",
    "user": "pass",
    "": ""
}

# Static list of models with their corresponding processing functions
MODEL_LIST = [
    ("不加载模型", None),
    ("YOLO11_PT", "load_yolo11_pt_model"),
    ("YOLO11_NCNN", "load_yolo11_ncnn_model"),
    ("HaarCascades", "load_haarcascades_model"),
    ("YOLO_FastestV2_NCNN", "load_yolo_fastestv2_model")
]

TRACK_HISTORY = defaultdict(lambda: [])

class OutputRedirector:
    def __init__(self, text_edit, max_lines=1000):
        self.text_edit = text_edit
        self.max_lines = max_lines
        self.text_edit.setDocument(self.text_edit.document())
        self.text_edit.document().setMaximumBlockCount(self.max_lines)


    def write(self, text):
        # 将文本写入 QTextEdit 控件
        self.text_edit.append(text)
        self.text_edit.verticalScrollBar().setValue(self.text_edit.verticalScrollBar().maximum())

    def flush(self):
        # 必须实现 flush 方法
        pass

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)
    inference_time_updated = pyqtSignal(float)
    detect_count_updated = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.running = True
        self.video = None
        self.fps_tick = time.time()
        self.fps = 0.0
        self.inference_time = 0.0
        self.current_model = None
        self.current_model_name = "不加载模型"
        self.detect_count = 0
        self.enable_tracking = False

    def run(self):
        self.connect_camera()
        while self.running:
            ret, frame = self.video.read()
            if not ret:
                frame = np.zeros((height, width, 3), np.uint8)
                self.video.release()
                print("摄像头已掉线，正在尝试重新连接")
                self.connect_camera()

            if ret:
                # Run inference if a model is selected
                if self.current_model:
                    inference_start = time.time()
                    frame = self.run_inference(frame)
                    self.inference_time = (time.time() - inference_start) * 0.3 + self.inference_time * 0.7
                self.frame_ready.emit(frame)
                current_time = time.time()
                self.fps = (1.0 / (current_time - self.fps_tick)) * 0.2 + self.fps * 0.8
                self.fps_tick = current_time
                self.fps_updated.emit(self.fps)
                self.inference_time_updated.emit(self.inference_time)
                self.detect_count_updated.emit(self.detect_count)

    def connect_camera(self):
        video_devices = glob.glob('/dev/video*')
        for device in video_devices:
            print(f"正在检测 {device}")
            self.video = cv2.VideoCapture(device)
            if self.video.isOpened():
                self.video.set(3, width)
                self.video.set(4, height)
                ret, _ = self.video.read()
                if ret:
                    print(f"成功连接到 {device}")
                    return
        print("没有可用的摄像头设备")

    def run_inference(self, frame):
        if self.current_model_name == "YOLO11_PT":
            if self.enable_tracking:
                frame = self.run_yolo11_pt_inference_track(frame)
            else:
                frame = self.run_yolo11_pt_inference(frame)
        elif self.current_model_name == "YOLO11_NCNN":
            if self.enable_tracking:
                frame = self.run_yolo11_ncnn_inference_track(frame)
            else:
                frame = self.run_yolo11_ncnn_inference(frame)
        elif self.current_model_name == "HaarCascades":
            frame = self.run_haarcascades_inference(frame)
        elif self.current_model_name == "YOLO_FastestV2_NCNN":
            frame = self.run_yolo_fastestv2_inference(frame)
        return frame

    def run_yolo11_pt_inference(self, frame):
        results = self.current_model(frame)
        frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        idx = 0
        for box in boxes:
            x, y, w, h = box
            print(f"检测到人脸: (X:{x:.2f}, Y:{y:.2f}, 宽:{w:.2f}, 高:{h:.2f}), 可信度: {confs[idx]:.2f}")
            idx += 1
        self.detect_count = idx
        return frame

    def run_yolo11_pt_inference_track(self, frame):
        results = self.current_model.track(frame, persist=True)
        frame = results[0].plot()
        idx = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = TRACK_HISTORY[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 15:  # retain 90 tracks for 90 frames
                    track.pop(0)
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
                print(f"检测到人脸: (X:{x:.2f}, Y:{y:.2f}, 宽:{w:.2f}, 高:{h:.2f}), 可信度: {confs[idx]:.2f}")
                idx += 1
        self.detect_count = idx
        return frame

    def run_yolo11_ncnn_inference(self, frame):
        results = self.current_model(frame, imgsz=export_imgsz)
        frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        idx = 0
        for box in boxes:
            x, y, w, h = box
            print(f"检测到人脸: (X:{x:.2f}, Y:{y:.2f}, 宽:{w:.2f}, 高:{h:.2f}), 可信度: {confs[idx]:.2f}")
            idx += 1
        return frame

    def run_yolo11_ncnn_inference_track(self, frame):
        results = self.current_model.track(frame, imgsz=export_imgsz, persist=True)
        frame = results[0].plot()
        idx = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = TRACK_HISTORY[track_id]
                track.append((float(x), float(y)))
                if len(track) > 15:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
                print(f"检测到人脸: (X:{x:.2f}, Y:{y:.2f}, 宽:{w:.2f}, 高:{h:.2f}), 可信度: {confs[idx]:.2f}")
                idx += 1
        return frame

    def run_haarcascades_inference(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.current_model.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(f"检测到人脸: (X:{x:.2f}, Y:{y:.2f}, 宽:{w:.2f}, 高:{h:.2f})")
        self.detect_count = len(faces)
        return frame

    def run_yolo_fastestv2_inference(self, frame):
        results = self.current_model.infer(frame)
        idx = 0
        for box in results or []:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            w = x2 - x1
            h = y2 - y1
            label = f'{box["class"]}: {box["score"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 17), (x1 + 100, y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            print(f"检测到人脸: (X:{x1:.2f}, Y:{y1:.2f}, 宽:{w:.2f}, 高:{h:.2f}), 可信度: {box['score']:.2f}")
            idx += 1
        self.detect_count = idx
        return frame

    def stop(self):
        self.running = False
        if self.video:
            self.video.release()
        self.quit()
        self.wait()


class LoginApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("登陆界面")
        self.resize(300, 150)

        self.username_label = QLabel("账号:", self)
        self.password_label = QLabel("密码:", self)
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("登陆", self)
        self.login_button.clicked.connect(self.authenticate)

        layout = QVBoxLayout()
        form_layout = QHBoxLayout()
        form_layout.addWidget(self.username_label)
        form_layout.addWidget(self.username_input)
        layout.addLayout(form_layout)

        form_layout = QHBoxLayout()
        form_layout.addWidget(self.password_label)
        form_layout.addWidget(self.password_input)
        layout.addLayout(form_layout)

        layout.addWidget(self.login_button)
        self.setLayout(layout)

    def authenticate(self):
        username = self.username_input.text()
        password = self.password_input.text()
        if username in USER_DATABASE and USER_DATABASE[username] == password:
            QMessageBox.information(self, "成功", "欢迎进入系统")
            self.open_camera_app()
        else:
            QMessageBox.critical(self, "失败", "账号或密码错误")

    def open_camera_app(self):
        self.close()
        self.camera_app = CameraApp()
        self.camera_app.show()


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.capture_thread = VideoCaptureThread()
        self.capture_thread.frame_ready.connect(self.update_frame)
        self.capture_thread.fps_updated.connect(self.update_fps)
        self.capture_thread.inference_time_updated.connect(self.update_inference_time)
        self.capture_thread.detect_count_updated.connect(self.detect_count_updated)
        self.capture_thread.start()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("正在初始化摄像头")

        # 右侧区域的标签、复选框和按钮
        self.fps_label = QLabel("当前帧率: 0 FPS", self)
        self.fps_label.setAlignment(Qt.AlignLeft)

        self.inference_time_label = QLabel("推理用时: 0 ms", self)
        self.inference_time_label.setAlignment(Qt.AlignLeft)

        self.detect_count_label = QLabel("检测到人脸: 0", self)
        self.detect_count_label.setAlignment(Qt.AlignLeft)

        self.choose_model_label = QLabel("选择模型:", self)
        self.choose_model_label.setAlignment(Qt.AlignLeft)

        self.enable_tracking_checkbox = QCheckBox("开启跟踪", self)
        self.enable_tracking_checkbox.stateChanged.connect(self.on_tracking_checkbox_changed)

        self.quit_button = QPushButton("退出系统", self)
        self.quit_button.clicked.connect(self.close_app)

        self.model_select = QComboBox(self)
        for model_name, _ in MODEL_LIST:
            self.model_select.addItem(model_name)
        self.model_select.currentIndexChanged.connect(self.on_model_change)

        # 创建 QTextEdit 控件
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)  # 设置为只读

        # 左侧区域 (包含 image_label)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)

        # 右侧区域 (包含 fps, inference_time, model select 等)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.fps_label)
        right_layout.addWidget(self.inference_time_label)
        right_layout.addWidget(self.detect_count_label)
        right_layout.addWidget(self.choose_model_label)
        right_layout.addWidget(self.model_select)
        right_layout.addWidget(self.enable_tracking_checkbox)
        right_layout.addWidget(self.quit_button)

        # 主布局 (包含左区域和右区域)
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)  # 左边部分
        main_layout.addLayout(right_layout, 1)  # 右边部分

        # 添加 QTextEdit 到布局底部
        bottom_layout = QVBoxLayout()
        bottom_layout.addLayout(main_layout)
        bottom_layout.addWidget(self.text_edit)

        self.setLayout(bottom_layout)
        self.setWindowTitle("人脸检测系统")
        self.resize(540, 380)  # 根据实际需要调整窗口大小

        self.redirector = OutputRedirector(self.text_edit)
        sys.stdout = self.redirector

    def on_tracking_checkbox_changed(self, state):
        if state == Qt.Checked:
            self.capture_thread.enable_tracking = True
        else:
            self.capture_thread.enable_tracking = False

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def update_fps(self, fps):
        self.fps_label.setText(f"当前帧率:{fps:.2f}FPS")

    def update_inference_time(self, inference_time):
        self.inference_time_label.setText(f"推理用时:{inference_time*1000:.2f}ms")

    def detect_count_updated(self, detect_count):
        self.detect_count_label.setText(f"检测到人脸: {detect_count}")

    def load_model(self):
        selected_model_name = self.model_select.currentText()
        if selected_model_name == "不加载模型":
            self.capture_thread.current_model = None
        else:
            # Find the corresponding function for the selected model
            for model_name, function_name in MODEL_LIST:
                if model_name == selected_model_name:
                    if function_name:
                        model_function = globals().get(function_name)
                        if model_function:
                            self.capture_thread.current_model = model_function()
                            self.capture_thread.current_model_name = selected_model_name
                            print(f"正在加载模型：{selected_model_name}")
                            break

    def on_model_change(self, index):
        # Update the model when the user changes the selection
        self.load_model()

    def close_app(self):
        self.capture_thread.stop()
        self.close()

    def closeEvent(self, event):
        self.capture_thread.stop()
        event.accept()


# Placeholder model load functions
def load_yolo11_pt_model():
    return YOLO("yolo11_128.pt")

def load_yolo11_ncnn_model():
    return YOLO("yolo11_128_ncnn_model")

def load_haarcascades_model():
    return cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

def load_yolo_fastestv2_model():
    arch = platform.machine()
    return YoloFastestV2("./yolo_fastestv2/yolo_fastestv2_" + arch)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginApp()
    login_window.show()
    sys.exit(app.exec_())
