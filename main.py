from PySide6.QtGui import QFontDatabase, QFont, QCursor, QPainter, QPen, QColor
from PySide6.QtWidgets import QLabel, QWidget, QTextEdit, QComboBox, QMessageBox, QApplication, QProgressBar, QPushButton
from PySide6.QtUiTools import QUiLoader
import random
from CamOperation_class import CameraOperation
from MvCameraControl_class import *
from datetime import datetime
from utils import save_capture_to_db
from notify import SendTelegramThread
from PySide6.QtGui import QImage, QPixmap
from libs.sdk.CameraParams_header import *

import typing
try:
    from typing_extensions import Self as _Self
    if not hasattr(typing, "Self"):
        typing.Self = _Self
except Exception:
    pass
from ultralytics import YOLO
from PySide6.QtCore import QFile, QTimer, QProcess, Slot
import ctypes

import os, glob, shutil, datetime, cv2

CAP_DIR = "captures"
os.makedirs(CAP_DIR, exist_ok=True)

from collections import deque
# ...
DEFECT_POINTS = deque(maxlen=2000)  # (x, y, idx)
CAPTURE_IDX   = 0
HEAT_WINDOW   = 15
LAST_FRAME    = None

def show_zoomed_on_label(label_widget, frame_bgr, zoom=2.0):
    import cv2
    from PySide6.QtGui import QImage, QPixmap

    h, w = frame_bgr.shape[:2]
    crop_w, crop_h = int(w/zoom), int(h/zoom)
    cx, cy = w // 2, h // 2
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    x2, y2 = x1 + crop_w, y1 + crop_h
    crop = frame_bgr[y1:y2, x1:x2].copy()

    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
    pix  = QPixmap.fromImage(qimg).scaled(
        label_widget.width(), label_widget.height(),
        Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    label_widget.setScaledContents(False)
    label_widget.setAlignment(Qt.AlignCenter)
    label_widget.setPixmap(pix)
    return crop

def capture_snapshot_plain():
    global CAPTURE_IDX
    CAPTURE_IDX += 1
    global obj_cam_operation, isOpen, isGrabbing, label_4, label_6, model, CONF_TH, IOU_TH
    global LAST_FRAME, DEFECT_POINTS

    if not (isOpen and isGrabbing and obj_cam_operation):
        QMessageBox.warning(window, "Capture", "The camera is not running (Open/Start_grabbing).", QMessageBox.Ok)
        return

    from libs.sdk.MvErrorDefine_const import MV_OK
    ret = obj_cam_operation.Save_Bmp()
    if ret != MV_OK:
        QMessageBox.warning(window, "Capture", f"Save_Bmp failed (ret={ret})", QMessageBox.Ok)
        return

    bmp_list = glob.glob("*.bmp")
    if not bmp_list:
        QMessageBox.warning(window, "Capture", "BMP not found after Save_Bmp()", QMessageBox.Ok)
        return
    bmp_list.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    src = bmp_list[0]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(CAP_DIR, f"shot_{ts}.bmp")
    try:
        shutil.copy2(src, raw_path)
    except Exception:
        raw_path = src

    img = cv2.imread(raw_path)
    if img is None:
        QMessageBox.warning(window, "Capture", f"Failed to read image: {raw_path}", QMessageBox.Ok)
        return

    try:
        results = model(img, conf=CONF_TH, iou=IOU_TH, verbose=False)
        res = results[0]
        annotated, counts = yolo_annotate(img, res)
        det_path = os.path.join(CAP_DIR, f"shot_{ts}_det.jpg")
        cv2.imwrite(det_path, annotated)
        row_id, max_conf = save_capture_to_db(raw_path, det_path, counts, res)
        if counts.get("defect", 0) > 0:
            caption = (f"❌ DEFECT DETECTED\nid:{row_id} defects:{counts['defect']} "
                       f"samples:{counts.get('sample',0)} max_conf:{max_conf:.2f}")
            SendTelegramThread(det_path, caption).start()

        LAST_FRAME = annotated.copy()
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names
            for (x1, y1, x2, y2), cid in zip(xyxy, clss):
                cname = str(names.get(int(cid), cid)).lower()
                if cname in ("defect", "defects", "ng", "bad"):
                    cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                    DEFECT_POINTS.append((cx, cy, CAPTURE_IDX))
    except Exception as e:
        print("YOLO skipped/error:", e)
        counts = {"defect": 0}

    ZOOM = 2.0
    zoom_crop = show_zoomed_on_label(label_4, img, zoom=ZOOM)

    zoom_path = os.path.join(CAP_DIR, f"shot_{ts}_zoom.jpg")
    cv2.imwrite(zoom_path, zoom_crop)

    # label_6.setText("❌" if counts.get("defect", 0) > 0 else "✅")

    print("📸 Saved raw:", raw_path)
    print("🔎 Saved zoom:", zoom_path)


def capture_snapshot():
    import os, glob, shutil, datetime
    import numpy as np
    import cv2
    from PySide6.QtGui import QImage, QPixmap

    global CAPTURE_IDX
    CAPTURE_IDX += 1

    global obj_cam_operation, isOpen, isGrabbing, label_4, label_6, model, CONF_TH, IOU_TH
    global LAST_FRAME, DEFECT_POINTS

    if not (isOpen and isGrabbing and obj_cam_operation):
        QMessageBox.warning(window, "Capture", "The camera is not running (Open/Start_grabbing).", QMessageBox.Ok)
        return

    from libs.sdk.MvErrorDefine_const import MV_OK
    ret = obj_cam_operation.Save_Bmp()
    if ret != MV_OK:
        QMessageBox.warning(window, "Capture", f"Save_Bmp failed (ret={ret})", QMessageBox.Ok)
        return
    print("[A] Save_Bmp ok")

    bmp_list = glob.glob("*.bmp")
    if not bmp_list:
        QMessageBox.warning(window, "Capture", "BMP not found after Save_Bmp()", QMessageBox.Ok)
        return
    bmp_list.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    src = bmp_list[0]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(CAP_DIR, f"shot_{ts}.bmp")
    try:
        shutil.copy2(src, raw_path)
    except Exception as e:
        print(f"[B] copy2 failed: {e} → we use the source")
        raw_path = src
    print(f"[B] picked {raw_path}")

    img = cv2.imread(raw_path, cv2.IMREAD_COLOR)
    if img is None:
        QMessageBox.warning(window, "Capture", f"Failed to read image: {raw_path}", QMessageBox.Ok)
        return
    print("[C] imread ok", img.shape)

    try:
        print("[D] YOLO start")
        results = model(img, conf=CONF_TH, iou=IOU_TH, verbose=False)
        res = results[0]
        annotated, counts = yolo_annotate(img, res)
        print("[E] YOLO done")
    except Exception as e:
        QMessageBox.warning(window, "YOLO", f"Inference error: {e}", QMessageBox.Ok)
        return

    det_path = os.path.join(CAP_DIR, f"shot_{ts}_det.jpg")
    cv2.imwrite(det_path, annotated)
    print("[F] saved det", det_path)

    try:
        row_id, max_conf = save_capture_to_db(raw_path, det_path, counts, res)
        if counts.get("defect", 0) > 0:
            caption = (f"❌ DEFECT DETECTED\n"
                       f"id: {row_id}\n"
                       f"defects: {counts['defect']} | samples: {counts.get('sample', 0)}\n"
                       f"max_conf: {max_conf:.2f}\n"
                       f"{os.path.basename(det_path)}")
            SendTelegramThread(det_path, caption).start()
    except Exception as e:
        print("[DB/TG] skipped due error:", e)

    LAST_FRAME = annotated.copy()
    try:
        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy
            clss = res.boxes.cls
            xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            clss = clss.cpu().numpy().astype(int) if hasattr(clss, "cpu") else np.asarray(clss, dtype=int)
            names = res.names if hasattr(res, "names") else {}
            for (x1, y1, x2, y2), cid in zip(xyxy, clss):
                cname = str(names.get(int(cid), cid)).lower()
                if cname in ("defect", "defects", "ng", "bad"):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    DEFECT_POINTS.append((cx, cy, CAPTURE_IDX))
        print(f"[G] defect points stored = {len(DEFECT_POINTS)}")
    except Exception as e:
        print("[G] points error:", e)

    ZOOM = 2.0
    h, w = annotated.shape[:2]
    crop_w, crop_h = int(w / ZOOM), int(h / ZOOM)
    cx, cy = w // 2, h // 2
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    x2, y2 = x1 + crop_w, y1 + crop_h
    crop = annotated[y1:y2, x1:x2].copy()
    crop = np.ascontiguousarray(crop)
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                  rgb.strides[0], QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(qimg)
    label_4.setScaledContents(False)
    label_4.setAlignment(Qt.AlignCenter)
    scaled = pixmap.scaled(label_4.width(), label_4.height(),
                           Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label_4.setPixmap(scaled)
    print("[H] QImage->QPixmap ok; UI updated")

    label_6.setText("❌" if counts.get("defect", 0) > 0 else "✅")

    print("📸 Saved raw:", raw_path)
    print("🧠 YOLO det:", det_path)

import numpy as np
CLS_COLORS = {
    "sample": (255, 200, 0),
    "defect": (0, 0, 255),
}

def yolo_annotate(img_bgr, res):
    annotated = img_bgr.copy()
    names = res.names
    counts = {"sample": 0, "defect": 0, "other": 0}

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, counts

    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [None]*len(clss)

    for (x1, y1, x2, y2), cid, conf in zip(xyxy, clss, confs):
        cname = names.get(int(cid), str(cid)).lower()
        label = cname
        if conf is not None:
            label = f"{cname} {conf:.2f}"

        if cname in ("defect", "defects"):
            color = CLS_COLORS["defect"]
            counts["defect"] += 1
        elif cname in ("sample", "samples"):
            color = CLS_COLORS["sample"]
            counts["sample"] += 1
        else:
            color = (0, 255, 255)
            counts["other"] += 1

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty1 = max(y1 - th - 4, 0)
        cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 6, ty1 + th + 4), color, -1)
        cv2.putText(annotated, label, (x1 + 3, ty1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    legend = f"sample: {counts['sample']}  defect: {counts['defect']}"
    cv2.putText(annotated, legend, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

    return annotated, counts

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

PY32 = r"libs\Python313-32\python.exe"
DOBOT_SCRIPT = r"libs\dobot_sdk\rer.py"

if font_id == -1:
    print("Error")
else:
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]

loader = QUiLoader()
ui_file = QFile("app_design.ui")
ui_file.open(QFile.ReadOnly)
window = loader.load(ui_file)
QTimer.singleShot(0, lambda: enum_devices())

if window is None:
    raise RuntimeError("UI failed to load (loader.load returned None). Check the path to app_design.ui and custom widgets.")

from PySide6.QtCore import Qt
def must_find(name, cls):
    w = window.findChild(cls, name, Qt.FindChildrenRecursively)
    if w is None:
        names = [b.objectName() for b in window.findChildren(QPushButton, options=Qt.FindChildrenRecursively)]
        raise RuntimeError(f"Didn't find it {cls.__name__} with objectName='{name}'. The buttons were found: {names}")
    return w

dobot_start_btn = must_find("pushButton_4", QPushButton)
dobot_enter_btn = must_find("pushButton_5", QPushButton)
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
    global obj_cam_operation, isOpen, isGrabbing, deviceList

    if not isOpen:
        if deviceList is None or getattr(deviceList, "nDeviceNum", 0) == 0:
            enum_devices()
            if getattr(deviceList, "nDeviceNum", 0) == 0:
                QMessageBox.warning(window, "Error", "No devices found. Check cable/power or close MVS Viewer.", QMessageBox.Ok)
                return

        nSelCamIndex = comboBoxDevices.currentIndex()
        if nSelCamIndex < 0 or nSelCamIndex >= deviceList.nDeviceNum:
            QMessageBox.warning(window, "Error", f"Bad index: {nSelCamIndex}. Re-enumerate devices.", QMessageBox.Ok)
            return

        obj_cam_operation = CameraOperation(obj_cam, deviceList, nSelCamIndex)

        ret = obj_cam_operation.Open_device()
        if ret != 0:
            try:
                from libs.sdk.MvErrorDefine_const import MV_E_ACCESS, MV_E_BUSY, MV_E_RESOURCE
            except Exception:
                pass
            QMessageBox.warning(window, "Error", f"Open device failed! ret=0x{ret & 0xffffffff:08x}\n"
                                                 f"• Close MVS Viewer / other applications\n"
                                                 f"• Reconnect the camera and repeat Enum", QMessageBox.Ok)
            return

        isOpen = True

        ret = obj_cam_operation.Start_grabbing(label_2.winId())
        if ret != 0:
            QMessageBox.warning(window, "Error", f"Start grabbing failed! ret=0x{ret & 0xffffffff:08x}", QMessageBox.Ok)
            obj_cam_operation.Close_device()
            isOpen = False
            return

        isGrabbing = True
        start_button.setText("Stop")

    else:
        if obj_cam_operation:
            obj_cam_operation.Stop_grabbing()
            obj_cam_operation.Close_device()

        isOpen = False
        isGrabbing = False
        start_button.setText("開始：開啟")
        QMessageBox.information(window, "Info", "Camera stopped successfully", QMessageBox.Ok)

        label_2.setStyleSheet("""
            QLabel {
                background-image: url("media/pictures/main_photo.png");
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
        background-image: url("media/pictures/main_photo.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid white;
	    border-color:  #f8f44c;
    }
""")

label_5.setStyleSheet("""
    QLabel {
        background-image: url("media/pictures/radar.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid #f8f44c;
    }
""")

label_4.setStyleSheet("""
    QLabel {
        background-image: url("media/pictures/zoom.png");
        background-repeat: no-repeat;
        background-position: center;
        background-color: transparent;
        border: 1px solid #f8f44c;
    }
""")

label.setText(f'''
    <span style="font-family: '{font_family}'; font-size: 42px; color: white;">
        AlphaMold
        <img src="media/pictures/logo.png" height="50" style="vertical-align: top; margin-left: 10px;">
    </span>
''')

ZOOM = 2.9

def show_heatmap_on_label(label_widget, base_bgr, points, alpha=0.45, zoom=ZOOM):

    if base_bgr is None:
        QMessageBox.information(window, "Heatmap", "No frame. Take a photo first.", QMessageBox.Ok)
        return

    h, w = base_bgr.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for p in list(points) if points is not None else []:
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            heat[y, x] += 1.0

    if heat.max() > 0:
        sigma = max(w, h) / 100.0
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
        heat_norm  = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        overlay    = cv2.addWeighted(base_bgr, 1 - alpha, heat_color, alpha, 0)
    else:
        overlay = base_bgr.copy()

    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    cx, cy = w // 2, h // 2
    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    x2, y2 = x1 + crop_w, y1 + crop_h
    crop = overlay[y1:y2, x1:x2].copy()

    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
    pix  = QPixmap.fromImage(qimg).scaled(
        label_widget.width(), label_widget.height(),
        Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    label_widget.setScaledContents(False)
    label_widget.setAlignment(Qt.AlignCenter)
    label_widget.setPixmap(pix)


label.setTextFormat(Qt.RichText)
label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
label_2.setScaledContents(True)

label.setFont(QFont(font_family, 32))
pixmap = QPixmap("media/pictures/logo.png")


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

label_7 = window.findChild(QLabel, "label_7")

pix = create_diagonal_stripe_pixmap(300, 100, QColor("#f8f44c"), QColor("#131313"))
# label_7.setPixmap(pix)
# label_7.setScaledContents(True)
# label_7.setStyleSheet("color: #131313; font-size: 1px;")
# label_7.setFixedHeight(40)

switch_btn = window.findChild(QPushButton, "pushButton")
is_on = False
switch_btn.setText("開始：關閉")

def toggle_state():
    global is_on
    is_on = not is_on
    switch_btn.setText("開始：開啟" if is_on else "開始：關閉")

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
        switch_btn_3.setText("掃描：關閉")
        global is_on_2
        is_on_2 = False

progress_timer = QTimer()
progress_timer.timeout.connect(update_progress)

def toggle_state_2():
    global is_on_2, progress_value
    is_on_2 = not is_on_2

    if is_on_2:
        switch_btn_3.setText("掃描：開啟")
        progress_value = 0
        progress_bar.setValue(0)
        progress_timer.start(40)
    else:
        switch_btn_3.setText("掃描：關閉")
        progress_timer.stop()
        progress_bar.setValue(0)

# switch_btn_3.clicked.connect(toggle_state_2)
terminal_timer = QTimer()

terminal_timer.timeout.connect(update_progress)
start_button.clicked.connect(lambda: terminal_timer.start(400))

cursor_normal  = QPixmap("media/app/cursor.png").scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)
# cursor_click = QCursor(QPixmap("app\GPM\Design\cursor_2.png").scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation))

cursor = QCursor(cursor_normal, 0, 0)

window.setCursor(cursor)
window.menuBar().setCursor(cursor)
for child in window.findChildren(QWidget):
    child.setCursor(cursor)

# terminal_box = window.findChild(QTextEdit, "textEdit")
# sys.stdout = EmittingStream(terminal_box)
# sys.stderr = EmittingStream(terminal_box)


# def on_press():
#     window.setCursor(cursor_click)
#
# def on_release():
#     window.setCursor(cursor_normal)
#
# button.pressed.connect(on_press)
# button.released.connect(on_release)

terminal_box = window.findChild(QTextEdit, "textEdit")
start_button = window.findChild(QPushButton, "pushButton")

dobot_start_btn = window.findChild(QPushButton, "pushButton_4")
dobot_enter_btn = window.findChild(QPushButton, "pushButton_5")

dobot_proc = QProcess(window)
dobot_proc.setProcessChannelMode(QProcess.MergedChannels)  # stdout+stderr

from pathlib import Path
dobot_proc.setWorkingDirectory(str(Path(DOBOT_SCRIPT).parent))

def log_append(msg: str):
    if terminal_box:
        terminal_box.append(msg.rstrip())
        terminal_box.ensureCursorVisible()
    else:
        print(msg)

@Slot()
def dobot_on_ready_read():
    try:
        data = bytes(dobot_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            log_append(data)
        err = bytes(dobot_proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if err:
            log_append(err)
    except Exception as e:
        log_append(f"[read error] {e}")

@Slot()
def dobot_on_started():
    log_append("[Dobot] 程序已啟動")
    dobot_start_btn.setEnabled(False)
    dobot_enter_btn.setEnabled(True)

@Slot(int, QProcess.ExitStatus)
def dobot_on_finished(code, status):
    log_append(f"[Dobot] 已完成 (code={code}, status={int(status)})")
    dobot_start_btn.setEnabled(True)
    dobot_enter_btn.setEnabled(False)

@Slot()
def dobot_start():
    if dobot_proc.state() != QProcess.NotRunning:
        log_append("[Dobot] 已在運行")
        return
    dobot_proc.start(PY32, [DOBOT_SCRIPT])

@Slot()
def dobot_send_enter():
    if dobot_proc.state() == QProcess.Running:
        dobot_proc.write(b"\n")
        dobot_proc.flush()
        log_append("[UI] ENTER → 已發送")
    else:
        log_append("[Dobot] 程序未啟動")

dobot_proc.readyReadStandardOutput.connect(dobot_on_ready_read)
dobot_proc.readyReadStandardError.connect(dobot_on_ready_read)
dobot_proc.started.connect(dobot_on_started)
dobot_proc.finished.connect(dobot_on_finished)

dobot_start_btn.clicked.connect(dobot_start)
dobot_enter_btn.clicked.connect(dobot_send_enter)
dobot_enter_btn.setEnabled(False)

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
        background-image: url("media/pictures/term_log.png");
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


heatmap_btn = window.findChild(QPushButton, "pushButton_3")

def on_heatmap_click():
    defect_count = len(DEFECT_POINTS)

    show_heatmap_on_label(label_5, LAST_FRAME, DEFECT_POINTS, alpha=0.45, zoom=ZOOM)

    if defect_count > 0:
        label_6.setText("❌")
        # label_6.setStyleSheet("QLabel { color: red; font-size: 90px; border: 1px solid #f8f44c; }")
    else:
        label_6.setText("✅")
        # label_6.setStyleSheet("QLabel { color: #6ee16e; font-size: 90px; border: 1px solid #f8f44c; }")

    DEFECT_POINTS.clear()

heatmap_btn.clicked.connect(on_heatmap_click)
start_button.clicked.connect(activate_terminal)


def scan_simulation():
    stop_grabbing()

    QMessageBox.information(window, "掃描完成", "掃描完成！", QMessageBox.Ok)

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
            start_btn.setText("開始：關閉")
        else:
            start_btn.setText("開始：開啟")
    else:
        start_btn.setText("開始：開啟")

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
# scan_btn.clicked.connect(scan_simulation)

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


# btn.clicked.connect(show_output_image)
# scan_btn.clicked.connect(capture_snapshot)
scan_btn.clicked.connect(capture_snapshot_plain)

window.showMaximized()
app.exec()