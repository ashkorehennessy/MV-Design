import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap

width = 320
height = 240

# Simple user database
USER_DATABASE = {
    "admin": "password123",
    "user": "pass"
}

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.video = None
        self.fps_tick = time.time()
        self.fps = 0.0

    def run(self):
        self.video = cv2.VideoCapture('/dev/video0')
        self.video.set(3, width)
        self.video.set(4, height)

        while self.running:
            ret, frame = self.video.read()
            if not ret:
                frame = np.zeros((height, width, 3), np.uint8)
                self.video.release()
                print("Camera disconnected. Attempting to reconnect...")

                for video_index in range(10):
                    time.sleep(1)
                    path = f'/dev/video{video_index}'
                    print(f"Testing {path}")
                    self.video = cv2.VideoCapture(path)
                    ret, frame = self.video.read()
                    if ret:
                        print("Camera reconnected.")
                        self.video.set(3, width)
                        self.video.set(4, height)
                        break

            if ret:
                self.frame_ready.emit(frame)
                current_time = time.time()
                self.fps = (1.0 / (current_time - self.fps_tick)) * 0.2 + self.fps * 0.8
                self.fps_tick = current_time
                self.fps_updated.emit(self.fps)

    def stop(self):
        self.running = False
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
            QMessageBox.information(self, "Login Successful", "Welcome!")
            self.open_camera_app()
        else:
            QMessageBox.critical(self, "Login Failed", "Invalid username or password.")

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

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.quit_button)
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

    def close_app(self):
        self.capture_thread.stop()
        self.close()

    def closeEvent(self, event):
        self.capture_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = LoginApp()
    login_window.show()
    sys.exit(app.exec_())