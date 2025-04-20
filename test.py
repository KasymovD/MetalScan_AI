from PySide6.QtWidgets import QPushButton
import os

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
