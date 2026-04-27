"""
Модуль 4: Обробка результатів та вивід
Точно та красиво представити результати розпізнавання емоцій
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

"""
Модуль для обробки та представлення результатів розпізнавання емоцій,
включає функції для форматування результатів, візуалізації на зображенні, 
генерації текстових звітів та збереження результатів.
"""

# Клас для класифікації результатів
class ResultHandler:
    # Словник кольорів під емоції для візуалізації
    EMOTION_COLORS = {
        'Angry': (0, 0, 255),        # Червоний
        'Disgust': (0, 165, 255),    # Оранжевий
        'Fear': (128, 0, 128),       # Фіолетовий
        'Happy': (0, 255, 0),        # Зелений
        'Neutral': (200, 200, 200),  # Сірий
        'Sad': (255, 0, 0),          # Синій
        'Surprise': (0, 255, 255)    # Жовтий
    }
    
    # Ініціалізація результатів
    def __init__(self):
        self.results = []
        self.last_report = None
    
    # Функція для форматування результату емоції
    def format_emotion_result(self, emotion: str, confidence: float) -> str:
        """
        Форматувати результат емоції для подальшого виведення
        
        Args:
            emotion (str): Назва емоції
            confidence (float): Впевненість (0-100)
            
        Returns:
            str: Форматований рядок ("Happy: 92.5%")
        """
        # Перевіряємо, чи емоція є валідною
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        
        # Обмежуємо впевненість до діапазону [0, 100]
        confidence = max(0, min(100, confidence))
        return f"{emotion}: {confidence:.1f}%"
    
    # Функція для візуалізації результатів на зображенні
    def visualize_results_on_image(self, image, face_coords: Tuple, 
                                   emotion: str, confidence: float) -> np.ndarray:
        """
        Намалювати результати прямо на зображенні
        - Намалювати прямокутник навколо обличчя
        - Написати назву емоції та впевненість
        
        Args:
            image (np.ndarray): Вхідне зображення
            face_coords (tuple): Координати обличчя (x, y, w, h)
            emotion (str): Розпізнана емоція
            confidence (float): Впевненість (0-100)
            
        Returns:
            np.ndarray: Зображення з результатами
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            return image
        
        # Копіюємо зображення для візуалізації результатів
        result = image.copy()
        # Розпаковуємо координати обличчя
        x, y, w, h = face_coords
        
        # Отримуємо колір для емоції
        color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # 1. Малюємо прямокутник навколо обличчя
        thickness = 3 # Товщина лінії
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness) # Малюємо прямокутник
        
        # 2. Створюємо фон для тексту 
        text_main = f"{emotion} ({confidence:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # Отримуємо розмір тексту
        text_size = cv2.getTextSize(text_main, font, font_scale, font_thickness)[0]
        
        # Розміри фону
        text_bg_padding = 10
        bg_x1 = x
        bg_y1 = y - text_size[1] - 2 * text_bg_padding # Фон над обличчям
        bg_x2 = x + text_size[0] + 2 * text_bg_padding # Ширина фону з урахуванням відступів
        bg_y2 = y
        
        # Малюємо фон
        cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # 3. Пишемо текст
        text_color = (255, 255, 255)  # Білий текст
        text_x = x + text_bg_padding # Відступ зліва
        text_y = y - text_bg_padding # Відступ знизу від верхньої межі фону
        
        cv2.putText(result, text_main, (text_x, text_y), 
                   font, font_scale, text_color, font_thickness)
        
        return result
    
    # Функція для витягування потрібних ділянок та  візуалізації
    def visualize_multiple_faces(self, image, faces_data: List[Dict]) -> np.ndarray:
        """
        Намалювати результати для кількох облич
        
        Args:
            image (np.ndarray): Вхідне зображення
            faces_data (list): Список словників з даними облич
                [{
                    'coords': (x, y, w, h),
                    'emotion': str,
                    'confidence': float
                }, ...]
            
        Returns:
            np.ndarray: Зображення з результатами для всіх облич
        """
        result = image.copy()
        
        # Проходимо по кожному обличчю та візуалізуємо результат
        for face_data in faces_data:
            result = self.visualize_results_on_image(
                result,
                face_data['coords'],
                face_data['emotion'],
                face_data['confidence']
            )
        
        return result
    
    # Функція для побудови текстового графіку ймовірностей емоцій
    def plot_emotion_probabilities(self, probabilities: Dict[str, float]) -> str:
        """
        Побудувати текстовий графік ймовірностей емоцій
        
        Args:
            probabilities (dict): {емоція: ймовірність (0-1)}
            
        Returns:
            str: ASCII art графік
        """
        result = "\n📊 Ймовірності емоцій:\n"
        result += "=" * 50 + "\n"
        
        # Сортуємо за ймовірністю (від найбільшої до найменшої)
        sorted_emotions = sorted(probabilities.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        
        # Знаходимо максимальну ймовірність для нормалізації графіку
        max_prob = max(probabilities.values()) if probabilities else 1.0
        
        # Побудова графіку
        for emotion, prob in sorted_emotions:
            # Нормалізуємо для графіку
            bar_length = int((prob / max_prob) * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            result += f"{emotion:12} {prob*100:6.2f}% [{bar}]\n"
        
        result += "=" * 50 + "\n"
        
        return result
    
    # Функція для генерації текстового звіту результатів
    def generate_report(self, image_path: str, face_count: int, 
                       emotion: str, confidence: float,
                       all_probabilities: Dict[str, float]) -> str:
        """
        Генерувати текстовий звіт результатів
        
        Args:
            image_path (str): Шлях до оригінального зображення
            face_count (int): Кількість виявлених облич
            emotion (str): Розпізнана емоція
            confidence (float): Впевненість
            all_probabilities (dict): Ймовірності для всіх емоцій
            
        Returns:
            str: Текстовий звіт
        """
        # Формуємо звіт у вигляді тексту з основною інформацією та графіком ймовірностей
        report = []
        report.append("\n" + "=" * 60)
        report.append("📋 ЗВІТ ПРО РОЗПІЗНАВАННЯ ЕМОЦІЙ")
        report.append("=" * 60)
        
        # Основна інформація
        report.append(f"\n📁 Файл: {image_path}")
        report.append(f"⏰ Час аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n👤 Обличчя виявлено: {face_count}")
        
        # Результат для найбільшого обличчя
        if face_count > 0:
            report.append(f"😊 Розпізнана емоція: {emotion}")
            report.append(f"📊 Впевненість: {confidence:.2f}%")
        else:
            report.append("⚠️  Облич не виявлено!")
        
        # Всі ймовірності
        if all_probabilities:
            report.append("\n" + self.plot_emotion_probabilities(all_probabilities))

        report.append("=" * 60 + "\n")
        
        report_text = "\n".join(report)
        self.last_report = report_text
        return report_text
    
    # Функція для збереження зображення з результатами
    def save_result_image(self, image, output_path: str) -> bool:
        """
        Зберегти зображення з результатами
        
        Args:
            image (np.ndarray): Зображення з результатами
            output_path (str): Шлях для збереження
            
        Returns:
            bool: True якщо успішно, False якщо помилка
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Недійсне зображення")
            
            # Переконуємось, що директорія існує
            import os
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Зберігаємо
            success = cv2.imwrite(output_path, image)
            
            if success:
                print(f"Результат збережено: {output_path}")
                return True
            else:
                print(f"Не вдалося зберегти: {output_path}")
                return False
                
        except Exception as e:
            print(f"❌ Помилка при збереженні: {e}")
            return False
    
    # Функція для виведення результатів у консоль
    def display_results_summary(self, results: Dict) -> None:
        """
        Показати підсумок результатів у консолі
        
        Args:
            results (dict): Словник з результатами:
                {
                    'image_path': str,
                    'face_count': int,
                    'faces': [
                        {
                            'emotion': str,
                            'confidence': float,
                            'probabilities': dict
                        },
                        ...
                    ]
                }
        """
        print("\n" + "=" * 60)
        print("✨ РЕЗУЛЬТАТИ АНАЛІЗУ")
        print("=" * 60)
        
        print(f"\n📁 Файл: {results.get('image_path', 'Unknown')}")
        print(f"👤 Облич виявлено: {results.get('face_count', 0)}")
        
        faces = results.get('faces', [])
        # Виводимо інформацію для кожного обличчя
        if faces:
            # Виводимо деталі для кожного обличчя
            for i, face_data in enumerate(faces, 1):
                emotion = face_data.get('emotion', 'Unknown')
                confidence = face_data.get('confidence', 0)
                
                print(f"\n  [{i}] Обличчя:")
                print(f"      Емоція: {emotion}")
                print(f"      Впевненість: {confidence:.2f}%")
                
                # Показуємо топ-3 емоції
                probs = face_data.get('probabilities', {})
                if probs:
                    top_3 = sorted(probs.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:3]
                    print(f"      Топ-3:")
                    for emotion_name, prob in top_3:
                        print(f"        • {emotion_name}: {prob*100:.1f}%")
        else:
            print("\nОблич не виявлено на зображенні!")
        
        print("\n" + "=" * 60 + "\n")
    
    # Функція для додавання результату до історії
    def add_result(self, image_path: str, emotion: str, 
                   confidence: float, face_count: int) -> None:
        """
        Додати результат до історії
        
        Args:
            image_path (str): Шлях до файлу
            emotion (str): Розпізнана емоція
            confidence (float): Впевненість
            face_count (int): Кількість облич
        """
        result_entry = {
            'timestamp': datetime.now(),
            'image_path': image_path,
            'emotion': emotion,
            'confidence': confidence,
            'face_count': face_count
        }
        self.results.append(result_entry)
    
    # Функція для отримання історії результатів
    def get_results_history(self) -> List[Dict]:
        """
        Отримати історію результатів
        
        Returns:
            list: Список всіх результатів
        """
        return self.results.copy()
    
    # Функція для очищення історії результатів
    def clear_history(self) -> None:
        """Очистити історію результатів"""
        self.results.clear()
        print("✅ Історія результатів очищена")


# Ініціалізація глобального обробника
result_handler = ResultHandler()
