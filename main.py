from PyQt5.QtWidgets import *
from PySide6.QtGui import QFontDatabase, QFont, QCursor, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel, QWidget, QTextEdit, QComboBox, QMessageBox, QApplication, QProgressBar, QPushButton
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QTimer
import random
from CamOperation_class import CameraOperation
from MvCameraControl_class import *
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import io
from ultralytics import YOLO

app = QApplication([])

model = YOLO("model/best.pt")
CONF_TH = 0.4
IOU_TH  = 0.5

font_id = QFontDatabase.addApplicationFont("media/fonts/Chopsic.otf")
MvCamera.MV_CC_Initialize()
deviceList = MV_CC_DEVICE_INFO_LIST()
cam = MvCamera()
obj_cam_operation = None
isOpen = False
isGrabbing = False

if font_id == -1:
    print("Error")
else:
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]

loader = QUiLoader()
ui_file = QFile("app_design.ui")
ui_file.open(QFile.ReadOnly)
window = loader.load(ui_file)
ui_file.close()

obj_cam = MvCamera()
deviceList = MV_CC_DEVICE_INFO_LIST()

isOpen = False
isGrabbing = False

obj_cam_operation = CameraOperation(obj_cam, deviceList)
deviceList = MV_CC_DEVICE_INFO_LIST()

connect_button = window.findChild(QPushButton, "connect_button")
comboBoxDevices = window.findChild(QComboBox, "comboBoxDevices")

label = window.findChild(QLabel, "label")
label_2 = window.findChild(QLabel, "label_2")
label_5 = window.findChild(QLabel, "label_5")
label_4 = window.findChild(QLabel, "label_4")

label_6 = window.findChild(QLabel, "label_6")

start_button = window.findChild(QPushButton, "pushButton")

def decoding_char(c_ubyte_value):
    c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        decode_str = c_char_p_value.value.decode('gbk')
    except UnicodeDecodeError:
        decode_str = str(c_char_p_value.value)
    return decode_str

def start_stop_camera():
    global obj_cam_operation, isOpen, isGrabbing

    if not isOpen:
        nSelCamIndex = comboBoxDevices.currentIndex()
        if nSelCamIndex < 0:
            QMessageBox.warning(window, "Error", "Please select a camera", QMessageBox.Ok)
            return

        obj_cam_operation = CameraOperation(obj_cam, deviceList, nSelCamIndex)

        ret = obj_cam_operation.Open_device()
        if ret != 0:
            QMessageBox.warning(window, "Error", "Open device failed!", QMessageBox.Ok)
            return

        isOpen = True

        ret = obj_cam_operation.Start_grabbing(label_2.winId())
        if ret != 0:
            QMessageBox.warning(window, "Error", "Start grabbing failed!", QMessageBox.Ok)
            return

        isGrabbing = True
        start_button.setText("Stop")
        # QMessageBox.information(window, "Info", "Camera started successfully", QMessageBox.Ok)

    else:
        if obj_cam_operation:
            obj_cam_operation.Stop_grabbing()
            obj_cam_operation.Close_device()

        isOpen = False
        isGrabbing = False
        start_button.setText("Start")
        QMessageBox.information(window, "Info", "Camera stopped successfully", QMessageBox.Ok)

        label_2.setStyleSheet("""
            QLabel {
                background-image: url("media/photos/main_photo.png");
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
                border: 1px solid white;
                border-color: #f8f44c;
            }
        """)


def enum_devices():
    global deviceList
    deviceList = MV_CC_DEVICE_INFO_LIST()
    n_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE)

    ret = MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)
    if ret != 0:
        QMessageBox.warning(window, "Error", "Enum devices fail!", QMessageBox.Ok)
        return

    if deviceList.nDeviceNum == 0:
        QMessageBox.warning(window, "Info", "No devices found", QMessageBox.Ok)
        return

    devList = []
    for i in range(deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName)
            model_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
            devList.append(f"[{i}] {user_defined_name} {model_name}")

    comboBoxDevices.clear()
    comboBoxDevices.addItems(devList)
    comboBoxDevices.setCurrentIndex(0)

connect_button.clicked.connect(enum_devices)
start_button.clicked.connect(start_stop_camera)

label_2.setStyleSheet("""
    QLabel {
        background-image: url("media/photos/main_photo.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid white;
	    border-color:  #f8f44c;
    }
""")

label_5.setStyleSheet("""
    QLabel {
        background-image: url("media/photos/radar.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid #f8f44c;
    }
""")

label_4.setStyleSheet("""
    QLabel {
        background-image: url("media/photos/zoom.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid #f8f44c;
    }
""")

label.setText(f'''
    <span style="font-family: '{font_family}'; font-size: 42px; color: white;">
        MetalScan
        <img src="media/photos/logo.png" height="50" style="vertical-align: top; margin-left: 10px;">
    </span>
''')

label.setTextFormat(Qt.RichText)
label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
label_2.setScaledContents(True)

label.setFont(QFont(font_family, 32))
pixmap = QPixmap("media/photos/logo.png")


def create_diagonal_stripe_pixmap(width, height, stripe_color, bg_color):
    pixmap = QPixmap(width, height)
    pixmap.fill(bg_color)

    painter = QPainter(pixmap)
    pen = QPen(stripe_color, 3)
    painter.setPen(pen)
    step = 20
    for x in range(-height, width, step):
        painter.drawLine(x, 0, x + height, height)
    painter.end()
    return pixmap

label_7 = window.findChild(QLabel, "label_7") # установить

pix = create_diagonal_stripe_pixmap(300, 100, QColor("#f8f44c"), QColor("#131313"))
# label_7.setPixmap(pix)
# label_7.setScaledContents(True)
# label_7.setStyleSheet("color: #131313; font-size: 1px;")
# label_7.setFixedHeight(40)

switch_btn = window.findChild(QPushButton, "pushButton")
is_on = False
switch_btn.setText("Start: OFF")

def toggle_state():
    global is_on
    is_on = not is_on
    switch_btn.setText("Start: ON" if is_on else "Start: OFF")

switch_btn.clicked.connect(toggle_state)

switch_btn_3 = window.findChild(QPushButton, "pushButton_2")
progress_bar = window.findChild(QProgressBar, "progressBar")

is_on_2 = False
progress_value = 0
timer = QTimer()

def update_progress():
    global progress_value
    if progress_value < 100:
        progress_value += random.randint(5, 40)
        progress_value = min(progress_value, 100)
        progress_bar.setValue(progress_value)
    else:
        timer.stop()
        switch_btn_3.setText("Scan: OFF")
        global is_on_2
        is_on_2 = False

progress_timer = QTimer()
progress_timer.timeout.connect(update_progress)

def toggle_state_2():
    global is_on_2, progress_value
    is_on_2 = not is_on_2

    if is_on_2:
        switch_btn_3.setText("Scan: ON")
        progress_value = 0
        progress_bar.setValue(0)
        progress_timer.start(40)
    else:
        switch_btn_3.setText("Scan: OFF")
        progress_timer.stop()
        progress_bar.setValue(0)

switch_btn_3.clicked.connect(toggle_state_2)
terminal_timer = QTimer()

terminal_timer.timeout.connect(update_progress)
start_button.clicked.connect(lambda: terminal_timer.start(400))

cursor_normal  = QPixmap("media/app/cursor.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
# cursor_click = QCursor(QPixmap("app\GPM\Design\cursor_2.png").scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))

cursor = QCursor(cursor_normal, 0, 0)

window.setCursor(cursor)
window.menuBar().setCursor(cursor)  # Важно!F
for child in window.findChildren(QWidget):
    child.setCursor(cursor)

# terminal_box = window.findChild(QTextEdit, "textEdit")
# sys.stdout = EmittingStream(terminal_box)
# sys.stderr = EmittingStream(terminal_box)


# def on_press():
#     window.setCursor(cursor_click)
#
# # При отпускании — возвращаем обычный курсор
# def on_release():
#     window.setCursor(cursor_normal)
#
# button.pressed.connect(on_press)
# button.released.connect(on_release)

terminal_box = window.findChild(QTextEdit, "textEdit")
start_button = window.findChild(QPushButton, "pushButton")

def activate_terminal():
    terminal_box.setStyleSheet("""
        QTextEdit {
            background-color: #131313;
            color: magenta;
            border: 1px solid #f8f44c;
        }
    """)


terminal_box.setStyleSheet("""
    QTextEdit {
        background-image: url("media/photos/term_log.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid #f8f44c;
    }
""")

terminal_box.setFont(QFont("Courier New", 11))
terminal_box.setReadOnly(True)

messages = [
    "Reading package lists...",
    "Building dependency tree...",
    "Reading state information...",
    "Installing package...",
    "Unpacking files...",
    "Setting up dependencies...",
    "Starting service...",
    "Installation completed successfully",
]

messages += [
    "Detecting operating system version...",
    "Allocating disk space...",
    "Initializing setup environment...",
    "Connecting to update server...",
    "Dependency check passed.",
    "Extracting installation files...",
    "Creating user configuration...",
    "Applying default settings...",
    "Loading kernel modules...",
    "Generating cache files...",
    "System reboot may be required.",
    "Validating digital signatures...",
    "Downloading checksum files...",
    "Checksum verification complete.",
    "Preparing upgrade environment...",
    "Installation queued...",
    "Queuing package updates...",
    "Scanning installed packages...",
    "Analyzing disk usage...",
    "Setting environment variables...",
    "Synchronizing repositories...",
    "Installing language packs...",
    "Installation speed: Fast",
    "Waiting for user confirmation...",
    "Automatic retry in progress...",
    "Post-installation tasks running...",
    "Installation finished with no errors.",
    "Installation finished with warnings.",
    "Starting first-time setup wizard...",
    "Confirming license agreement...",
]

current_line = 0

def update_terminal_log():
    global current_line

    if current_line < len(messages):
        terminal_box.append(messages[current_line])
        terminal_box.ensureCursorVisible()
        current_line += 1
    else:
        current_line = 0
        terminal_box.append(">> Simulation complete.")

terminal_timer = QTimer()
terminal_timer.timeout.connect(update_terminal_log)


pushButton_4 = window.findChild(QPushButton, "pushButton_3")
label_5 = window.findChild(QLabel, "label_5")

def draw_radar_chart():
    # Категории и значения
    labels = ['Crack', 'Scratch', 'Rust', 'Shape', 'Deformation']
    values_1 = [0.3, 0.4, 0.5, 0.3, 0.6]
    values_2 = [0.3, 0.3, 0.3, 0.3, 0.3]

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values_1 += values_1[:1]
    values_2 += values_2[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#131313')
    ax.set_facecolor('#131313')

    ax.plot(angles, values_1, color='magenta', linewidth=2, label='Current')
    ax.plot(angles, values_2, color='#f8f44c', linewidth=2, label='Ideal')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], color='white')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white')

    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(loc='upper right', facecolor='#131313', edgecolor='white', labelcolor='white')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = QImage()
    img.loadFromData(buf.getvalue())
    pixmap = QPixmap.fromImage(img)
    buf.close()
    plt.close(fig)

    scaled_pixmap = pixmap.scaled(
        label_5.width(), label_5.height(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )

    label_5.setStyleSheet("""QLabel {
                background-repeat: no-repeat;
                background-position: center;
                background-color: transparent;
                border: 1px solid white;
                border-color: #f8f44c;
            }""")
    label_5.setPixmap(scaled_pixmap)
    label_5.setScaledContents(False)



pushButton_4.clicked.connect(draw_radar_chart)
start_button.clicked.connect(activate_terminal)


def scan_simulation():
    stop_grabbing()

    QMessageBox.information(window, "Scanning complete", "Scan Done !", QMessageBox.Ok)

    start_grabbing()

def stop_grabbing():
    global isGrabbing
    ret = obj_cam_operation.Stop_grabbing()
    if ret == 0:
        isGrabbing = False
        enable_controls()
        print("🔴 Screen Stopped")
    else:
        QMessageBox.warning(window, "Error", f"Stop grabbing failed ret: {ToHexStr(ret)}", QMessageBox.Ok)

def start_grabbing():
    global isGrabbing
    ret = obj_cam_operation.Start_grabbing(label_2.winId())
    if ret == 0:
        isGrabbing = True
        enable_controls()
        print("🟢 Screen started")
    else:
        QMessageBox.warning(window, "Error", f"Start grabbing failed ret: {ToHexStr(ret)}", QMessageBox.Ok)

def enable_controls():
    scan_btn = window.findChild(QPushButton, "pushButton_2")
    scan_btn.setEnabled(isOpen and isGrabbing)


    start_btn = window.findChild(QPushButton, "pushButton")
    if isOpen:
        if isGrabbing:
            start_btn.setText("Start: OFF")
        else:
            start_btn.setText("Start: ON")
    else:
        start_btn.setText("Start: ON")

def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

scan_btn = window.findChild(QPushButton, "pushButton_2")
scan_btn.clicked.connect(scan_simulation)

def show_output_image():
    image_path = "app/output.jpg"
    pixmap = QPixmap(image_path)

    if not pixmap.isNull():
        scaled = pixmap.scaled(
            label_4.width(), label_4.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label_4.setPixmap(scaled)
        label_4.setScaledContents(True)

        label_6.setText("❌")
        label_6.setStyleSheet("""
                    QLabel {
                        color: white;
                        font-weight: bold;
                        font-size: 90px;
                        border: 1px solid #f8f44c;
                    }
                """)
    else:
        print("⚠️ output.jpg not found")

btn = window.findChild(QPushButton, "pushButton_2")
label_4 = window.findChild(QLabel, "label_4")
label_6 = window.findChild(QLabel, "label_6")

btn.clicked.connect(show_output_image)

window.showMaximized()
app.exec()