# Імпортую необхідні бібліотеки, в основі використовується OpenCV для обробки зображень та виявлення облич, а також NumPy для роботи з масивами даних.
import cv2
import numpy as np
from typing import List, Tuple

"""
Модуль з виявленням обличчя,
передбачає використання каскадів Хаара для виявлення необхідної ділянки на зображенні,
а також додаткові функції для обробки та валідації виявлених облич.
"""

# Клас детекції облич
class FaceDetector:
    # Ініціалізація детектора та завантаження каскадів Хаара
    def __init__(self):
        """
         Ініціалізація каскадів Хаара для виявлення облич,
         каскади Хаара є стандартним інструментом в бібліотеці OpenCV, 
         готове рішення використовується для оптимізації програмного коду.
        """
        # Отримуємо шлях до каскадів Хаара з OpenCV
        cascade_path = cv2.data.haarcascades
        
        # Завантажуємо додаткові каскади для більшої точності
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Альтернативний каскад для більш точного виявлення
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        
        # Перевіряємо, чи каскади завантажилися
        if self.face_cascade.empty():
            raise RuntimeError("Не вдалося завантажити каскад для виявлення облич!")
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Основна функція виявлення обличчя на зображенні
        
        Args:
            image (np.ndarray): Зображення в відтінках сірого (Grayscale)
            scale_factor (float): Фактор масштабування для пірамід (1.01-1.4)
            min_neighbors (int): Мінімум сусідів для допустимого прямокутника (3-6)
            min_size (tuple): Мінімальний розмір виявленого обличчя
            
        Returns:
            list: Список координат виявлених облич [(x, y, w, h), ...]
            
        Example:
            >>> detector = FaceDetector()
            >>> faces = detector.detect_faces(gray_image)
            >>> print(f"Знайдено {len(faces)} облич")
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            return []
        
        # Перевіряємо, якщо зображення кольорове, конвертуємо в сіре
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Виявляємо обличчя
        faces = self.face_cascade.detectMultiScale(
            image, # Вхідне зображення
            scaleFactor=scale_factor, # Фактор масштабування для пірамід
            minNeighbors=min_neighbors, # Мінімум сусідів для допустимого прямокутника
            minSize=min_size, # Мінімальний розмір виявленого обличчя
            flags=cv2.CASCADE_SCALE_IMAGE # Флаг для масштабування зображення
        )
        
        return list(faces)
    
    # Функція для отримання області виявленого обличчя
    def get_face_regions(self, image, faces):
        """
        Області облич з зображення
        
        Args:
            image (np.ndarray): Вхідне зображення
            faces (list): Список координат облич [(x, y, w, h), ...]
            
        Returns:
            list: Список під-зображень облич
        """
        # Перевіряємо, чи є обличчя для обробки
        if not faces.any() if isinstance(faces, np.ndarray) else not faces:
            return []
        
        # Витягуємо області обличчя з зображення
        face_regions = []
        # Проходимо по кожному виявленому обличчю та витягуємо його область
        for (x, y, w, h) in faces:
            face_region = image[y:y+h, x:x+w]
            face_regions.append(face_region)
        
        return face_regions
    
    # Функція для малювання прямокутників навколо виявлених облич
    def draw_face_rectangles(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Функція для малювання прямокутників навколо виявлених облич
        
        Args:
            image (np.ndarray): Вхідне зображення
            faces (list): Список координат облич [(x, y, w, h), ...]
            color (tuple): RGB/BGR колір прямокутника
            thickness (int): Товщина лінії
            
        Returns:
            np.ndarray: Зображення з намальованими прямокутниками
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            return image
        
        result = image.copy()
        
        # Проходимо по кожному виявленому обличчю та малюємо прямокутник
        for (x, y, w, h) in faces:
            # Намалюємо прямокутник
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Додаємо текст "Face" над прямокутником
            cv2.putText(result, "Face", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result
    
    # Функція для виявлення найбільшого обличчя серед виявлених
    def get_largest_face(self, image, faces):
        """
        Функція отримання найбільшого обличчя з виявлених
        
        Args:
            image (np.ndarray): Вхідне зображення
            faces (list): Список координат облич [(x, y, w, h), ...]
            
        Returns:
            tuple: Координати найбільшого обличчя (x, y, w, h) або None
        """
        if not faces.any() if isinstance(faces, np.ndarray) else not faces:
            return None
        
        # Знаходимо найбільшу площу
        largest = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest)
    
    # Функція для валідації виявленого обличчя (перевірка розміру, яскравості тощо)
    def validate_face(self, face_region, min_size=20):
        """
        Перевірити валідність виявленого обличчя
        
        Args:
            face_region (np.ndarray): Область обличчя
            min_size (int): Мінімальний розмір у пікселях
            
        Returns:
            bool: True якщо обличчя валідне
        """
        # Перевіряємо, чи область обличчя дійсна
        if face_region is None or face_region.size == 0:
            return False
        
        # Отримуємо розміри області обличчя
        height, width = face_region.shape[:2]
        
        # Перевіряємо розмір
        if height < min_size or width < min_size:
            return False
        
        # Перевіряємо, чи не є зображення занадто темним або світлим
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_region
        
        # Обчислюємо середню яскравість
        mean_brightness = np.mean(gray)
        
        # Яскравість має бути в розумному діапазоні (30-225)
        if mean_brightness < 30 or mean_brightness > 225:
            return False
        
        return True
    
    # Функція для виявлення та валідації облич на зображенні
    def detect_and_validate_faces(self, image, scale_factor=1.1, 
                                  min_neighbors=5, min_size=(30, 30)):
        """
        Функція виявленяя облич та перевірити їх валідність
        
        Args:
            image (np.ndarray): Зображення
            scale_factor (float): Масштабний фактор
            min_neighbors (int): Мінімум сусідів
            min_size (tuple): Мінімальний розмір
            
        Returns:
            list: Список валідних облич та їх координати
        """
        # Виявляємо обличчя
        faces = self.detect_faces(image, scale_factor, min_neighbors, min_size)
        # Отримуємо області обличчя для валідації
        face_regions = self.get_face_regions(image, faces)
        
        # Перевіряємо валідність кожного виявленого обличчя та збираємо лише валідні
        valid_faces = []
        for (x, y, w, h), face_region in zip(faces, face_regions):
            if self.validate_face(face_region):
                valid_faces.append((x, y, w, h))
        
        return valid_faces


# Ініціалізація глобального детектора
face_detector = FaceDetector()
