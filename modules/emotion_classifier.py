"""
Імпорт необхідних бібліотек для функціонування, 
зокрема для обробки зображень (OpenCV), роботи з моделлю (TensorFlow/Keras) та обробки даних (NumPy),
також для типізації та роботи з файловою системою (os).
"""
import numpy as np
import os
from typing import Tuple, Dict
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

"""
Модуль класифікації емоцій, котрий розпізнає емоційний стан по виявленній області обличчя,
використовується модель MobileNetV2, навченна на датасеті RAF-DB,
який містить 7 класів емоцій: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
"""

# Клас розпізнавання емоцій
class EmotionClassifier:
    # Емоційні класи, які розпізнає модель
    EMOTION_CLASSES = [
        'Angry', 'Disgust', 'Fear',
        'Happy', 'Neutral', 'Sad', 'Surprise'
    ]

    # Розмір вхідних даних для моделі (160x160 для MobileNetV2)
    INPUT_SIZE = (160, 160)

    """
        Ініціалізація класифікатора емоцій, можна передати шлях до моделі для завантаження.
        Якщо шлях не заданий, модель можна буде завантажити пізніше за допомогою методу load_model().
    """
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.emotion_classes = self.EMOTION_CLASSES

    # Функція завантаження моделі з файлу
    def load_model(self):
        """
        Завантажити модель з файлу, якщо шлях до моделі не заданий або файл не існує, буде викинута помилка.
         Важливо, щоб модель була збережена у форматі, сумісному з Keras (наприклад, .h5), 
         і відповідала архітектурі, яку очікує цей клас (MobileNetV2 з 7 вихідними класами).
        """
        # Перевіряємо, чи шлях до моделі заданий та чи файл існує
        if self.model_path is None:
            raise ValueError("Path не заданий")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не знайдена: {self.model_path}")

        # Вивід статусу завантаження моделі для користувача
        print(f"📦 Loading model: {self.model_path}")
        self.model = load_model(self.model_path)
        print("✅ Model loaded successfully")

        # Повертаємо завантажену модель для подальшого використання
        return self.model

    # Функція препроцесування області обличчя для аналізу 
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        # Перевіряємо, чи зображення обличчя дійсне
        if face_image is None or face_image.size == 0:
            raise ValueError("❌ Порожнє зображення")

        # Змінюємо розмір до вхідного розміру моделі
        resized = cv2.resize(face_image, self.INPUT_SIZE)

        # Якщо grayscale, тоді конвертуємо в RGB
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        # OpenCV використовує BGR, тому конвертуємо в RGB
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Нормалізуємо пікселі до діапазону [-1, 1] для MobileNetV2
        processed = preprocess_input(resized.astype(np.float32))

        # Додаємо додаткову вісь для batch (1, 160, 160, 3)
        processed = np.expand_dims(processed, axis=0)

        # Повертаємо оброблене зображення, готове для аналізу моделлю
        return processed

    # Попередня обробка та виведення найімовіршіної варіації
    def predict_emotion(self, face_image: np.ndarray) -> Tuple[str, float]:
        # Перевіряємо, чи модель завантажена
        if self.model is None:
            raise ValueError("Модель не завантажена")

        # Препроцесуємо зображення обличчя для аналізу моделлю
        processed = self.preprocess_face(face_image)

        # Попередня оброка
        preds = self.model.predict(processed, verbose=0)[0]

        # Отримуємо індекс класу з найбільшою ймовірністю та відповідну впевненість
        idx = np.argmax(preds)
        confidence = float(preds[idx]) * 100
        emotion = self.emotion_classes[idx]

        return emotion, confidence

    # Функція для отримання ймовірностей для всіх емоцій
    def predict_emotions_probabilities(self, face_image: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Модель не завантажена")

        # Препроцесуємо зображення обличчя для аналізу моделлю
        processed = self.preprocess_face(face_image)

        # Отримуємо ймовірності для всіх класів емоцій
        preds = self.model.predict(processed, verbose=0)[0]

        return {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_classes, preds)
        }

    # Функція для візуалізації передбачення на зображенні
    def draw_prediction(self, image: np.ndarray, emotion: str, confidence: float):
        text = f"{emotion}: {confidence:.1f}%"
        cv2.putText(
            image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        return image


# Ініціалізуємо класифікатор емоцій для подальшого використання в інших модулях або тестах
emotion_classifier = EmotionClassifier()