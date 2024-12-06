import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, \
    QMessageBox, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

width = 160
height = 120
ncnn_model_imgsz = 128

# Simple user database
USER_DATABASE = {
    "admin": "password123",
    "user": "pass",
    "": ""
}

# Static list of models with their corresponding processing functions
MODEL_LIST = [
    ("No Model", None),
    ("YOLO11_PT", "load_yolo11_pt_model"),
    ("YOLO11_NCNN", "load_yolo11_ncnn_model")
]


class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.video = None
        self.fps_tick = time.time()
        self.fps = 0.0
        self.current_model = None
        self.current_model_name = "No Model"

    def run(self):
        self.connect_camera()
        while self.running:
            ret, frame = self.video.read()
            if not ret:
                frame = np.zeros((height, width, 3), np.uint8)
                self.video.release()
                print("Camera disconnected. Attempting to reconnect...")
                self.connect_camera()

            if ret:
                # Run inference if a model is selected
                if self.current_model:
                    frame = self.run_inference(frame)
                self.frame_ready.emit(frame)
                current_time = time.time()
                self.fps = (1.0 / (current_time - self.fps_tick)) * 0.2 + self.fps * 0.8
                self.fps_tick = current_time
                self.fps_updated.emit(self.fps)

    def connect_camera(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            print("Failed to connect to camera.")
            return
        self.video.set(3, width)
        self.video.set(4, height)

    def run_inference(self, frame):
        if self.current_model_name == "YOLO11_PT":
            frame = self.run_yolo11_pt_inference(frame)
        elif self.current_model_name == "YOLO11_NCNN":
            frame = self.run_yolo11_ncnn_inference(frame)
        return frame

    def run_yolo11_pt_inference(self, frame):
        # Load and run YOLOv3 model on frame
        # (You should implement the actual YOLOv3 inference here)
        # Placeholder for bounding box drawing
        results = self.current_model(frame)
        frame = results[0].plot()
        return frame

    def run_yolo11_ncnn_inference(self, frame):
        # Load and run SSD model on frame
        # (You should implement the actual SSD inference here)
        # Placeholder for bounding box drawing
        results = self.current_model(frame, imgsz=ncnn_model_imgsz, int8=True)
        frame = results[0].plot()
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
        self.setWindowTitle("Login")
        self.resize(300, 150)

        self.username_label = QLabel("Username:", self)
        self.password_label = QLabel("Password:", self)
        self.username_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Login", self)
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
            QMessageBox.information(self, "Success", "Welcome!")
            self.open_camera_app()
        else:
            QMessageBox.critical(self, "Failed", "Invalid username or password.")

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
        self.capture_thread.start()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Camera loading...")

        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setAlignment(Qt.AlignLeft)

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)

        self.model_select = QComboBox(self)
        for model_name, _ in MODEL_LIST:
            self.model_select.addItem(model_name)
        self.model_select.currentIndexChanged.connect(self.on_model_change)

        self.load_button = QPushButton("Load Model", self)
        self.load_button.clicked.connect(self.load_model)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.quit_button)
        layout.addWidget(self.model_select)
        layout.addWidget(self.load_button)

        self.setLayout(layout)
        self.setWindowTitle("Camera Reconnect Example")
        self.resize(800, 600)

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    @pyqtSlot(float)
    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def load_model(self):
        selected_model_name = self.model_select.currentText()
        if selected_model_name == "No Model":
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
                            print(f"Model {selected_model_name} loaded.")
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
    # Load YOLO model here
    model = YOLO("yolo11n.pt")
    return model


def load_yolo11_ncnn_model():
    # Load SSD model here
    model = YOLO("yolo11n_ncnn_model")
    return model


if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginApp()
    login_window.show()
    sys.exit(app.exec_())
