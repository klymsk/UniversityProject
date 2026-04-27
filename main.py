import sys
from PyQt6.QtWidgets import QApplication
from gui.ui import EmotionRecognitionApp

"""
Головна точка входу програми
"""

def main():
    # Запуск застосунку
    app = QApplication(sys.argv)
    window = EmotionRecognitionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()