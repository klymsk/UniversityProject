# імпорт основних бібліотек, зокрема PyQt6 для графічного зображення

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QTabWidget,
    QTextEdit, QProgressBar, QMessageBox, QSpinBox, QCheckBox,
    QComboBox, QGroupBox, QGridLayout
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
import cv2
import numpy as np
from pathlib import Path

"""
GUI інтерфейс для графічної демонстрації функціоналу 
та взаємодії з користувачем
"""

class EmotionRecognitionApp(QMainWindow):
    
    def __init__(self):
        # Ініціалізація UI
        super().__init__()
        self.current_image_path = None
        self.current_image = None
        self.processing_thread = None
        self.init_ui()
    
    def init_ui(self):
        # Загальні. параметри вікна графічного інтерфейсу
        self.setWindowTitle("🎭 Розпізнавання емоцій за зображенням") # Назва вікна
        self.setGeometry(100, 100, 1400, 900) # Статичні розміри вікна
        
        # Центральний віджет та лейаут
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Ліва частина панелі управління
        """
        Передбачає собою керування загальними процесами, якими є
        вибір зображення, налаштування параметрів обробки,
        запуск аналізу, відображення статусу моделі та результату.
        """
        left_panel = QVBoxLayout()
        
        # Заголовок
        title_label = QLabel("📁 Керування")
        title_font = QFont()
        title_font.setPointSize(14) # Розмір шрифту
        title_font.setBold(True) # Стиль шрифту
        title_label.setFont(title_font)
        left_panel.addWidget(title_label) # Додаємо заголовок до панелі
        
        # Перша кнопка у віджеті, вибір зображення
        self.select_btn = QPushButton("📂 Обрати зображення")
        # Графічна робота з кнопкою, встановлення розмірності, стилів та додавання до віджету
        self.select_btn.setMinimumHeight(40)
        btn_font = QFont()
        btn_font.setPointSize(11)
        self.select_btn.setFont(btn_font)
        self.select_btn.clicked.connect(self.select_image)
        left_panel.addWidget(self.select_btn)
        
        # Показуємо вибраний файл
        self.file_label = QLabel("Файл не обраний")
        self.file_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self.file_label)
        
        # Наступний логічний блок з вибором параметрів обробки
        params_group = QGroupBox("⚙️ Параметри обробки")
        params_layout = QGridLayout()
        
        """
        В параметрах обробки передбачено вибір фільтрів,
        видалення шуму, згладжування та регулювання експозиції. 
        """

        # Видалення шуму
        params_layout.addWidget(QLabel("Видалення шуму:"), 0, 0)
        self.denoise_check = QCheckBox("Увімкнено")
        self.denoise_check.setChecked(True) # Встановлюємо флажок на вибір
        params_layout.addWidget(self.denoise_check, 0, 1)
        
        # Згладжування
        params_layout.addWidget(QLabel("Згладжування:"), 1, 0)
        self.smooth_check = QCheckBox("Увімкнено")
        self.smooth_check.setChecked(True) # Встановлюємо флажок на вибір
        params_layout.addWidget(self.smooth_check, 1, 1)
        
        # Регульювання експозиції
        params_layout.addWidget(QLabel("Експозиція:"), 2, 0)
        self.exposure_check = QCheckBox("Увімкнено")
        self.exposure_check.setChecked(True) # Встановлюємо флажок на вибір
        params_layout.addWidget(self.exposure_check, 2, 1)
        
        params_group.setLayout(params_layout)
        left_panel.addWidget(params_group) # Додаємо параметри до віджету
        
        # Наступний елемент віджету це параметри детекції обличчя
        detector_group = QGroupBox("👤 Детектор облич")
        detector_layout = QGridLayout()
        
        # Параметри детекції обличчя: масштабний фактор та мінімум сусідів
        detector_layout.addWidget(QLabel("Scale Factor:"), 0, 0)
        self.scale_factor_spin = QSpinBox()
        self.scale_factor_spin.setRange(101, 130)
        self.scale_factor_spin.setValue(110) # Значення по замовчуванню
        self.scale_factor_spin.setSuffix(" %")
        detector_layout.addWidget(self.scale_factor_spin, 0, 1)
        
        detector_layout.addWidget(QLabel("Min Neighbors:"), 1, 0)
        self.min_neighbors_spin = QSpinBox()
        self.min_neighbors_spin.setRange(1, 10)
        self.min_neighbors_spin.setValue(5) # Значення по замовчуванню
        detector_layout.addWidget(self.min_neighbors_spin, 1, 1)
        
        detector_group.setLayout(detector_layout)
        left_panel.addWidget(detector_group) # Додаємо до віджету
        
        # Кнопка для запуску аналізу емоційного стану
        self.process_btn = QPushButton("▶️ Аналізувати емоції")
        # Графічна робота з кнопкою, встановлення розмірності, стилів та додавання до віджету
        self.process_btn.setMinimumHeight(40)
        process_font = QFont()
        process_font.setPointSize(11)
        self.process_btn.setFont(process_font)
        self.process_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        left_panel.addWidget(self.process_btn)
        
        # Прогресбар для відображення прогресу обробки
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)
        
        # Статус моделі, котрий демонструє чи використовуєтсья попередньо натренерована модель
        model_status_box = QGroupBox("🤖 Статус моделі")
        model_status_layout = QVBoxLayout()
        
        # Перевіряємо, чи модель існує
        from pathlib import Path
        model_path = Path("models/emotion_model.h5")
        
        # Якщо модель готова та перебуває в відповідній папці
        if model_path.exists():
            model_status = "✅ Натренована модель \nвиявлена!\nРезультати точні!"
            model_color = "green"
            status_text = "Модель готова до використання"
        # Якщо модель відсутня, попереджаємо користувача про випадкові результати та необхідність тренування
        else:
            model_status = "⚠️ Модель НЕ натренована!\nРезультати ВИПАДКОВІ!\n\nДля натренування:\n1. Завантажте Дата-сет\n2. Запустіть:\npython3 train_emotion_model.py"
            model_color = "orange"
            status_text = "Потрібне натренування"
        
        self.model_status_label = QLabel(model_status)
        self.model_status_label.setStyleSheet(f"color: {model_color}; font-weight: bold; font-size: 10px;")
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_status_layout.addWidget(self.model_status_label)
        
        model_status_box.setLayout(model_status_layout)
        left_panel.addWidget(model_status_box)
        
        # Статус обробки та результату, який оновлюється під час процесу та після завершення аналізу
        self.status_label = QLabel(status_text)
        status_color = "green" if model_color == "green" else "orange"
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        left_panel.addWidget(self.status_label)
        
        # Розповнювальний простір
        left_panel.addStretch()
        
        # Ініціалізація лівої панелі та додавання до основного лейауту
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(300)
        main_layout.addWidget(left_widget)
        
        # Права частина панелі

        """
        Передбачає собою відображення попереднього перегляду зображення,
        графічний результат аналізу та додаткові відомості.
        """
        tabs = QTabWidget()
        
        # Перший таб для превью зображення та результатів аналізу
        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        
        # Превью зображення
        self.preview_label = QLabel()
        self.preview_label.setText("📷 Превью зображення буде тут")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet(
            "border: 2px dashed gray; background-color: #f0f0f0;"
        )
        result_layout.addWidget(self.preview_label)
        
        # Результати
        self.results_text = QTextEdit()
        # Встановлення стилів для виклику результатів
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(250)
        result_layout.addWidget(QLabel("📊 Результати:"))
        result_layout.addWidget(self.results_text)
        
        tabs.addTab(result_tab, "👁️ Превью && Результати")
        
        # Другий таб для детальної інформації та звіту про аналіз
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        tabs.addTab(info_tab, "📋 Детальна інформація")
        
        # Третій таб для інформації про програму, архітектуру та інструкції
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setText("""
🎭 Розпізнавання емоцій за зображенням

📌 Версія: 1.0
📅 2026 рік

📋 Опис:
Програма використовує глибоку нейронну мережу
для розпізнавання емоційного стану людини за аналізом зображення.

🏗️ Архітектура:
1. Image Processor - попередня обробка зображень
2. Face Detector - виявлення облич (Haar Cascades)
3. Emotion Classifier - класифікація емоцій (CNN на основі MobileNetV2)
4. Result Handler - обробка та вивід результатів

📊 Емоції (7 класів):
😡 Angry - Гнів
🤢 Disgust - Відраза
😨 Fear - Страх
😊 Happy - Щастя
😐 Neutral - Нейтральна
😢 Sad - Сум
😲 Surprise - Здивування

🔧 Технологія:
- Python 3.9+
- OpenCV для обробки та вилення
- TensorFlow/Keras для машинного навчання
- PyQt6 для GUI

📚 Датасет:
RAF-DB (Real-world Affective Faces Database) - ~15,000 зображень з 7 класами емоцій
        """)
        about_layout.addWidget(about_text)
        
        tabs.addTab(about_tab, "ℹ️ Про програму")
        
        main_layout.addWidget(tabs)
        
        # Встановлюємо пропорції
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)
    
    # Кнопка вибору зображення
    def select_image(self):
        # Вибір зображення з діалогового вікна
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Обрати зображення",
            "",
            "Зображення (*.jpg *.jpeg *.png *.bmp);;Всі файли (*.*)"
        )
        
        if file_path and Path(file_path).exists():
            self.current_image_path = file_path
            self.file_label.setText(f"✅ {Path(file_path).name}")
            self.file_label.setStyleSheet("color: green;")
            self.process_btn.setEnabled(True)
            self.status_label.setText("Файл обраний. Натисніть 'Аналізувати'")
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")
            
            # Показуємо превью
            self.display_preview(file_path)
    
    # Функція для демонстрації превью обраного зображення
    def display_preview(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.show_error_message("Не вдалося завантажити зображення!")
                return
            
            # Конвертуємо BGR в RGB, оскільки OpenCV використовує BGR, а PyQt очікує RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Масштабуємо, щоб вмістилось в діапазон попереднього перегляду
            height, width = image_rgb.shape[:2]
            max_size = 400

            # Зберігаємо пропорції зображення
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            resized = cv2.resize(image_rgb, (new_width, new_height))
            
            # Конвертуємо в QPixmap для PyQt6
            h, w, ch = resized.shape
            bytes_per_line = 3 * w
            qt_image = QImage(resized.data, w, h, bytes_per_line, 
                            QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setText("")  # Чистимо текст
            
        except Exception as e:
            self.show_error_message(f"Помилка при завантаженні превью: {e}")
    
    # Кнопка для запуску аналізу зображення
    def process_image(self):
        # Перевірка чи вибрано зображення перед запуском аналізу
        if not self.current_image_path:
            self.show_error_message("Спочатку оберіть зображення!")
            return
        
        # Вимикаємо кнопку під час обробки
        self.process_btn.setEnabled(False)
        self.status_label.setText("⏳ Обробка...") # Відображення статусу обробки
        # Встановлення стилів та позиціонування
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.info_text.clear()
        
        # Запускаємо обробку в окремому потоці
        self.processing_thread = ImageProcessingThread(
            self.current_image_path,
            denoise = self.denoise_check.isChecked(), # Вибір застосування фільтра видалення шуму
            smooth = self.smooth_check.isChecked(), # Вибір застосування фільтра згладжування
            exposure = self.exposure_check.isChecked(), # Вибір застосування фільтра регулювання експозиції
            scale_factor = self.scale_factor_spin.value() / 100.0, # Вибір масштабного фактора для детекції обличчя
            min_neighbors = self.min_neighbors_spin.value() # Вибір мінімуму сусідів для детекції обличчя
        )
        
        self.processing_thread.progress.connect(self.on_progress) # Підключення сигналу для оновлення прогресу
        self.processing_thread.finished.connect(self.on_processing_finished) # Підключення сигналу для обробки результатів після завершення
        self.processing_thread.error.connect(self.on_processing_error) # Підключення сигналу для обробки помилок під час обробки
        
        self.processing_thread.start()
    
    # Функції для обробки сигналів від потока обробки зображення
    def on_progress(self, message):
        """Обновлення прогресу обробки"""
        self.status_label.setText(message)
        self.progress_bar.setValue(min(self.progress_bar.value() + 25, 90))
    
    # Функція для обробки результатів після завершення аналізу
    def on_processing_finished(self, results):
        """Обробка завершена - показати результати"""
        self.progress_bar.setValue(100)
        
        # Показуємо результати в консолі та UI
        results_text = f"""
✅ Результат аналізу

📁 Файл: {Path(self.current_image_path).name}

👤 Облич виявлено: {results['face_count']}

"""
        
        if results['face_count'] > 0:
            for i, face_data in enumerate(results['faces'], 1):
                emotion = face_data['emotion']
                confidence = face_data['confidence']
                results_text += f"\n🔹 Обличчя {i}:"
                results_text += f"\n   Емоція: {emotion}"
                results_text += f"\n   Впевненість: {confidence:.1f}%\n"
        
        self.results_text.setText(results_text)
        
        # Детальна інформація
        self.info_text.setText(results.get('report', ''))
        
        # Показуємо результуюче зображення з нарисованими емоціями
        if 'result_image_path' in results:
            self.display_preview(results['result_image_path'])
        
        self.status_label.setText("✅ Готово!")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    # Функція для обробки помилок під час обробки зображення
    def on_processing_error(self, error_message):
        """Помилка під час обробки"""
        self.show_error_message(f"Помилка обробки:\n\n{error_message}")
        self.process_btn.setEnabled(True)
        self.status_label.setText("❌ Помилка!")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.progress_bar.setVisible(False)
    
    # Функція для показу повідомлення про помилку
    def show_error_message(self, error_text):
        """Показати повідомлення про помилку"""
        QMessageBox.critical(self, "❌ Помилка", error_text)


class ImageProcessingThread(QThread):
    # Сигнали для комунікації з основним потоком
    finished = pyqtSignal(dict)  # Результати готові
    error = pyqtSignal(str)      # Сталася помилка
    progress = pyqtSignal(str)   # Повідомлення про прогрес
    
    # Ініціалізація потока з параметрами для обробки зображення та детекції обличчя
    def __init__(self, image_path, denoise=True, smooth=True, exposure=True,
                 scale_factor=1.1, min_neighbors=5):
        super().__init__()
        self.image_path = image_path
        self.denoise = denoise
        self.smooth = smooth
        self.exposure = exposure
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
    
    # Основна функція, яка виконується при запуску потока
    def run(self):
        try:
            # Імпортуємо модулі тільки під час обробки
            from modules.image_processor import ImagePreprocessor
            from modules.face_detector import FaceDetector
            from modules.emotion_classifier import EmotionClassifier
            from modules.result_handler import ResultHandler
            
            # Крок 1: Попередня обробка зображення
            self.progress.emit("📥 Завантаження та обробка зображення...")
            preprocessor = ImagePreprocessor()
            # Виконуємо повну попередню обробку зображення з вибраними фільтрами
            orig_image, processed_image, gray_image = preprocessor.preprocess_complete(
                self.image_path,
                apply_noise_removal = self.denoise, # Вибір застосування фільтра видалення шуму
                apply_smoothing = self.smooth, # Вибір застосування фільтра згладжування
                apply_exposure = self.exposure # Вибір застосування фільтра регулювання експозиції
            )
            
            # Крок 2: Виявлення облич
            self.progress.emit("👤 Виявлення облич...")
            detector = FaceDetector()
            faces = detector.detect_faces(
                gray_image,
                scale_factor=self.scale_factor,
                min_neighbors=self.min_neighbors
            )
            
            face_count = len(faces)
            result_image = processed_image.copy()
            
            # Крок 3: Класифікація емоцій для кожного обличчя
            self.progress.emit("😊 Розпізнавання емоцій...")
            
            # Зберігаємо інформацію про кожне обличчя
            faces_data = []
            
            if face_count > 0:
                # Якщо натренована модель існує, використовуємо її
                try:
                    classifier = EmotionClassifier(
                        model_path="models/emotion_model.h5"
                    )
                    classifier.load_model()
                    model_available = True
                except:
                    model_available = False
                    # Для демонстрації, генеруємо випадкові результати
                    import random
                    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                
                for i, (x, y, w, h) in enumerate(faces):
                    face_region = processed_image[y:y+h, x:x+w]
                    
                    # Використовуємо натреновану модель
                    if model_available:
                        emotion, confidence = classifier.predict_emotion(face_region)
                        probabilities = classifier.predict_emotions_probabilities(face_region) # Додатковий метод для отримання ймовірностей для всіх класів
                    # Для демонстрації без натренованої моделі
                    else:
                        emotion = random.choice(emotions)
                        confidence = random.uniform(50, 95)
                        probabilities = {e: random.random() for e in emotions}
                        # Нормалізуємо ймовірності
                        total = sum(probabilities.values())
                        probabilities = {e: p/total for e, p in probabilities.items()}
                    
                    # Додаємо дані про обличчя
                    faces_data.append({
                        'coords': (x, y, w, h), # Координати обличчя
                        'emotion': emotion, # Розпізнана емоція
                        'confidence': confidence, # Впевненість у розпізнаванні
                        'probabilities': probabilities # Ймовірності для всіх класів емоцій
                    })
            
            # Крок 4: Обробка та вивід результатів
            self.progress.emit("📊 Генерування результатів...")
            result_handler = ResultHandler()
            
            # Малюємо результати на зображенні
            result_with_vis = result_handler.visualize_multiple_faces(
                processed_image,
                faces_data
            )
            
            # Зберігаємо результат
            result_path = str(Path(self.image_path).parent / "result_emotion_analysis.jpg")
            result_handler.save_result_image(result_with_vis, result_path)
            
            # Генеруємо звіт
            if face_count > 0 and faces_data:
                main_emotion = faces_data[0]['emotion']
                main_confidence = faces_data[0]['confidence']
                main_probs = faces_data[0]['probabilities']
            else:
                main_emotion = "N/A"
                main_confidence = 0.0
                main_probs = {}
            
            report = result_handler.generate_report(
                self.image_path,
                face_count,
                main_emotion,
                main_confidence,
                main_probs
            )
            
            # Формуємо результат
            results = {
                'face_count': face_count,
                'faces': faces_data,
                'result_image_path': result_path,
                'report': report
            }
            
            # Відправляємо результат
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {str(e)}")
