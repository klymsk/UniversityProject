# Імпорт бібліотек для роботи з зображенням, основою є OpenCV та NumPy для обробки масивів зображень, а також pathlib для роботи з файловими шляхами.
import cv2
import numpy as np
from pathlib import Path

"""
Модуль для попередньої обробки обраного зображення,
передбачає завантаження зображення, видалення шумів, згладжування,
регулювання експозиції.

Використовує білатеральний фільтр для видалення шуму, Гаусівське розмиття для згладжування
та CLAHE для регулювання експозиції.
"""

# Клас попередньої обробки
class ImagePreprocessor:
    """
    Загалом існує три варіації вхідного зображення,
    оригінальне зображення (те, що завантажив користувач з ФС),
    оброблене зображення (після застосування фільтрів) та
    сіре зображення (конвертоване в відтінки сірого для детекції обличчя).

    Даний метод використовується для доступу до варіацій зображення в будь-який час.
    """
    # Ініціалізація
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.grayscale_image = None
    
    def load_image(self, image_path):
        """
            Функція завантажити зображення з файлу
            
            Args:
                image_path (str): Шлях до файлу зображення
                
            Returns:
                np.ndarray: Завантажене зображення (BGR формат!)
                
            Raises:
                FileNotFoundError: Якщо файл не існує
                ValueError: Якщо файл не є зображенням
        """
        # Перевіряємо, чи файл існує
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Файл не знайдено: {image_path}")
        
        # Перевіряємо, чи файл є зображенням (за розширенням)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            raise ValueError(f"Невірний формат файлу: {image_path}")
        
        # Завантажуємо зображення за допомогою OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не вдалося завантажити зображення: {image_path}")
        
        # Зберігаємо оригінальне зображення для подальшого використання
        self.original_image = image.copy()
        return image
    
    def remove_noise(self, image, method='bilateral'):
        """
            Видалення шумів з зображення
            
            Args:
                image (np.ndarray): Вхідне зображення
                method (str): Метод видалення шуму ('bilateral', 'gaussian', 'median')
                
            Returns:
                np.ndarray: Зображення без шуму
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            raise ValueError("Недійсне зображення")
        
        # Вибір методу видалення шуму з параметрів функції
        if method == 'bilateral':
            # Білатеральний фільтр
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            # Гаусівський фільтр
            denoised = cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            # Медіанний фільтр
            denoised = cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Невідомий метод: {method}")
        
        # Зберігаємо оброблене зображення для подальшого використання
        return denoised
    
    # Згладжування зображення
    def smooth_image(self, image, kernel_size=(5, 5)):
        """
        Згладити зображення Гаусівским розмиттям
        
        Args:
            image (np.ndarray): Вхідне зображення
            kernel_size (tuple): Розмір ядра (має бути непарним)
            
        Returns:
            np.ndarray: Згладжене зображення
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            raise ValueError("Недійсне зображення")
        
        # Переконуємось, що розмір ядра непарний
        k_size = (kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1,
                  kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1)
        
        # Застосовуємо Гаусівське розмиття
        smoothed = cv2.GaussianBlur(image, k_size, 0)
        return smoothed
    
    # Регулювання експозиції зображення
    def adjust_exposure(self, image):
        """
        Регулювати експозицію за допомогою CLAHE
        (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image (np.ndarray): Вхідне зображення (BGR!)
            
        Returns:
            np.ndarray: Зображення з відрегульованою експозицією
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            raise ValueError("Недійсне зображення")
        
        # Конвертуємо в HSV для роботи з яскравістю
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Застосовуємо CLAHE до каналу V (яскравості)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2].astype(np.uint8))
        
        # Конвертуємо назад в BGR
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return adjusted
    
    # Конвертація зображення в відтінки сірого
    def convert_to_grayscale(self, image):
        """
        Конвертувати зображення в відтінки сірого
        
        Args:
            image (np.ndarray): Вхідне зображення (BGR)
            
        Returns:
            np.ndarray: Зображення в сірих тонах (одноканальне)
        """
        # Перевіряємо, чи зображення дійсне
        if image is None or image.size == 0:
            raise ValueError("Недійсне зображення")
        
        # Використовуємо стандартне OpenCV перетворення BGR в Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.grayscale_image = gray.copy()

        # Зберігаємо сіре зображення для подальшого використання
        return gray
    
    # Повна попередня обробка зображення з обраними фільтрами
    def preprocess_complete(self, image_path, apply_noise_removal=True,
                           apply_smoothing=True, apply_exposure=True):
        """
        Повна попередня обробка зображення
        
        Args:
            image_path (str): Шлях до зображення
            apply_noise_removal (bool): Застосовувати видалення шуму
            apply_smoothing (bool): Застосовувати згладжування
            apply_exposure (bool): Застосовувати корекцію експозиції
            
        Returns:
            tuple: (оригінальне зображення, оброблене зображення, сіре зображення)
            
        Example:
            >>> preprocessor = ImagePreprocessor()
            >>> orig, processed, gray = preprocessor.preprocess_complete('photo.jpg')
        """
        # Завантажуємо зображення
        image = self.load_image(image_path)
        self.processed_image = image.copy()
        
        # Видаляємо шум
        if apply_noise_removal:
            self.processed_image = self.remove_noise(self.processed_image, method='bilateral')
        
        # Згладжуємо
        if apply_smoothing:
            self.processed_image = self.smooth_image(self.processed_image)
        
        # Регулюємо експозицію
        if apply_exposure:
            self.processed_image = self.adjust_exposure(self.processed_image)
        
        # Конвертуємо в відтінки сірого
        grayscale = self.convert_to_grayscale(self.processed_image)
        
        # Повертаємо всі три варіації зображення для подальшого використання
        return self.original_image, self.processed_image, grayscale


# Ініціалізація глобального процесора
preprocessor = ImagePreprocessor()
