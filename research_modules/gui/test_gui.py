import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import QTimer
from PIL import Image
import ipaddress
from base_station import BaseStation  # Import your BaseStation class here
import traceback

class BaseStationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.bs = BaseStation()  # Initialize BaseStation

        self.setWindowTitle("Base Station v0.1")
        self.setGeometry(100, 100, 640, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.canvas = QLabel(self.central_widget)
        self.canvas.setGeometry(0, 0, 640, 600)

        self.welcome_label = QLabel("Welcome on board, Captain Kurt!", self.canvas)
        self.ip_label = QLabel("Please enter boat's IP address to connect", self.canvas)
        self.ip_entry = QLineEdit(self.canvas)
        self.connect_button = QPushButton("Connect", self.canvas)
        self.connect_button.clicked.connect(self.on_connect)

        self.logo_label = QLabel(self.central_widget)
        self.mode_label = QLabel("Mode", self.central_widget)
        self.mode_switch = QComboBox(self.central_widget)
        self.mode_switch.addItems(["Auto", "Manual"])
        self.mode_switch.currentTextChanged.connect(self.switch_bs_mode)

        self.out_msg_label = QLabel("Sent:", self.central_widget)
        self.in_msg_label = QLabel("Received:", self.central_widget)

        self.shutdown_button = QPushButton("Shutdown", self.central_widget)
        self.shutdown_button.clicked.connect(self.on_shutdown)
        self.disconnect_button = QPushButton("Disconnect", self.central_widget)
        self.disconnect_button.clicked.connect(self.on_disconnect)

        self.video_label = QLabel(self.central_widget)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self.centralWidget())

        layout.addWidget(self.welcome_label)
        layout.addWidget(self.ip_label)
        layout.addWidget(self.ip_entry)
        layout.addWidget(self.connect_button)

        work_layout = QHBoxLayout()
        layout.addLayout(work_layout)

        work_layout.addWidget(self.logo_label)
        work_layout.addWidget(self.video_label)

        # Add mode selection and messages
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(self.mode_switch)
        work_layout.addLayout(mode_layout)

        messages_layout = QVBoxLayout()
        messages_layout.addWidget(self.out_msg_label)
        messages_layout.addWidget(self.in_msg_label)
        work_layout.addLayout(messages_layout)

        layout.addWidget(self.shutdown_button)
        layout.addWidget(self.disconnect_button)

    def switch_bs_mode(self, mode):
        if mode == "Auto":
            self.bs.mode = 'detect'
            self.bs.send_command("Auto mode")
        elif mode == "Manual":
            self.bs.mode = 'manual'
            self.bs.send_command("Manual mode")

    def on_connect(self):
        try:
            ip_address = str(ipaddress.ip_address(self.ip_entry.text()))
            print(f"Valid IP address: {ip_address}")
            self.bs.connect(host=ip_address, port=65432, in_msg_label=self.in_msg_label, out_msg_label=self.out_msg_label)
            self.show_start_screen(False)
            self.show_work_screen()
        except ValueError:
            QMessageBox.critical(self, "Unable to connect", f"No server is listening at '{self.ip_entry.text()}'")
        except Exception:
            QMessageBox.critical(self, "Error", "Couldn't initialize base station")
            traceback.print_exc()

    def on_disconnect(self):
        if QMessageBox.question(self, "Disconnect", "Do you want to disconnect and close the application?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.bs.send_command("Disconnect")
            self.bs.close()
            self.close()

    def on_shutdown(self):
        if QMessageBox.question(self, "Shutdown", "Do you want to shutdown the boat and close the application?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.bs.send_command("Shutdown")
            self.bs.close()
            self.close()

    def show_start_screen(self, is_displayed=True):
        self.canvas.setVisible(is_displayed)

    def show_work_screen(self, is_displayed=True):
        self.show_logo(is_displayed)
        self.show_boat_webcam(is_displayed)
        self.show_mode_selection(is_displayed)
        self.show_messages(is_displayed)

    def show_logo(self, is_displayed=True):
        if is_displayed:
            pixmap = QPixmap("config/logo_uet.png")
            self.logo_label.setPixmap(pixmap)
        else:
            self.logo_label.clear()

    def show_boat_webcam(self, is_displayed=True):
        # Implement the video streaming from the base station here
        pass

    def show_mode_selection(self, is_displayed=True):
        self.mode_label.setVisible(is_displayed)
        self.mode_switch.setVisible(is_displayed)

    def show_messages(self, is_displayed=True):
        self.out_msg_label.setVisible(is_displayed)
        self.in_msg_label.setVisible(is_displayed)

    def show_frame(self):
        ret, frame = self.bs.real_time_control()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)

        QTimer.singleShot(50, self.show_frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    base_station_app = BaseStationApp()
    base_station_app.show()
    base_station_app.show_frame()  # Start showing frames
    sys.exit(app.exec())
