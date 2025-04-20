import os
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

app = QApplication([])

# Загрузка шрифта
font_id = QFontDatabase.addApplicationFont("fonts/chopsic/Chopsic.otf")

if font_id == -1:
    print("❌ Ошибка загрузки шрифта")
else:
    font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
    print("✅ Шрифт загружен:", font_family)

# Загрузка UI
loader = QUiLoader()
ui_file = QFile("GMP.ui")
ui_file.open(QFile.ReadOnly)
window = loader.load(ui_file)
ui_file.close()

# Применение шрифта к QLabel
label = window.findChild(QLabel, "label")  # замените на ваш objectName
label.setFont(QFont(font_family, 16))  # 16 = размер

button = QPushButton("")

# Абсолютный путь
path = os.path.abspath("fonts/12.jpg").replace("\\", "/")

button.setStyleSheet(f"""
    QPushButton {{
        border: none;
        background-image: url("{path}");
        background-repeat: no-repeat;
        background-position: center;
    }}
""")


window.show()
app.exec()
