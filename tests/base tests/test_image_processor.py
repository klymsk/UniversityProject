"""
Тестовий скрипт для модуля Image Processor
Перевіряємо всі методи попередної обробки
"""

import cv2
import numpy as np
from modules.image_processor import ImagePreprocessor
from pathlib import Path


def test_image_processor():
    """Тестування ImagePreprocessor"""
    
    preprocessor = ImagePreprocessor()
    
    print("=" * 60)
    print("🧪 ТЕСТУВАННЯ IMAGE PROCESSOR")
    print("=" * 60)
    
    # Створюємо тестове зображення (якщо нема файлу)
    test_image_path = "test_image.jpg"
    
    # Перевіряємо, чи існує тестове зображення
    if not Path(test_image_path).exists():
        print(f"\n⚠️  Тестове зображення не знайдено: {test_image_path}")
        print("📝 Створюємо тестове зображення на льоту...")
        
        # Створюємо штучне зображення для тестування
        test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(test_image_path, test_img)
        print(f"✅ Тестове зображення створено: {test_image_path}")
    
    try:
        # Тест 1: Завантаження зображення
        print("\n[1/6] 📥 Тест завантаження зображення...")
        image = preprocessor.load_image(test_image_path)
        print(f"✅ Успішно завантажено. Розмір: {image.shape}")
        
        # Тест 2: Видалення шуму
        print("\n[2/6] 🔇 Тест видалення шуму...")
        denoised = preprocessor.remove_noise(image, method='bilateral')
        print(f"✅ Шум видалений. Метод: bilateral")
        
        # Тест 3: Згладжування
        print("\n[3/6] 🌊 Тест згладжування...")
        smoothed = preprocessor.smooth_image(denoised)
        print(f"✅ Зображення згладжено")
        
        # Тест 4: Регульювання експозиції
        print("\n[4/6] ☀️  Тест регульювання експозиції...")
        exposed = preprocessor.adjust_exposure(smoothed)
        print(f"✅ Експозиція відрегульована")
        
        # Тест 5: Конвертація в сіре
        print("\n[5/6] ⚫ Тест конвертації в сіре...")
        gray = preprocessor.convert_to_grayscale(exposed)
        print(f"✅ Зображення конвертовано в сіре. Розмір: {gray.shape}")
        
        # Тест 6: Повна обробка
        print("\n[6/6] 🔄 Тест повної обробки...")
        orig, processed, gray_full = preprocessor.preprocess_complete(
            test_image_path,
            apply_noise_removal=True,
            apply_smoothing=True,
            apply_exposure=True
        )
        print(f"✅ Повна обробка завершена")
        print(f"   - Оригінал: {orig.shape}")
        print(f"   - Оброблено: {processed.shape}")
        print(f"   - Сіре: {gray_full.shape}")
        
        # Збереження результатів
        print("\n📊 Збереження результатів обробки...")
        cv2.imwrite("output_original.jpg", orig)
        cv2.imwrite("output_processed.jpg", processed)
        cv2.imwrite("output_grayscale.jpg", gray_full)
        print("✅ Результати збережено (output_*.jpg)")
        
        print("\n" + "=" * 60)
        print("✅ ВСІ ТЕСТИ ПРОЙШЛИ УСПІШНО!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Помилка під час тестування: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_image_processor()
