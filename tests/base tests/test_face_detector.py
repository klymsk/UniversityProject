"""
Тестовий скрипт для модуля Face Detector
"""

import cv2
import numpy as np
from modules.image_processor import ImagePreprocessor
from modules.face_detector import FaceDetector


def test_face_detector():
    """Тестування FaceDetector"""
    
    print("=" * 60)
    print("🧪 ТЕСТУВАННЯ FACE DETECTOR")
    print("=" * 60)
    
    # Використовуємо тестове зображення, яке ми створили раніше
    test_image_path = "test_image.jpg"
    
    try:
        # Ініціалізуємо препроцесор та детектор
        preprocessor = ImagePreprocessor()
        detector = FaceDetector()
        
        print(f"\n[1/4] 📥 Завантаження та обробка зображення...")
        orig, processed, gray = preprocessor.preprocess_complete(test_image_path)
        print(f"✅ Зображення завантажено. Розмір: {gray.shape}")
        
        print(f"\n[2/4] 👤 Виявлення облич...")
        faces = detector.detect_faces(gray, scale_factor=1.05, min_neighbors=3)
        print(f"✅ Знайдено облич: {len(faces)}")
        
        if len(faces) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                print(f"   Обличчя {i+1}: x={x}, y={y}, w={w}, h={h}")
                
            print(f"\n[3/4] 📐 Витягування найбільшого обличчя...")
            largest = detector.get_largest_face(gray, faces)
            if largest:
                x, y, w, h = largest
                print(f"✅ Найбільше обличчя: x={x}, y={y}, w={w}, h={h}")
                print(f"   Розмір: {w}x{h} пікселів")
                
                # Витягуємо регіон обличчя
                face_region = gray[y:y+h, x:x+w]
                print(f"   Розмір вирізаного обличчя: {face_region.shape}")
                
                # Перевіряємо валідність
                is_valid = detector.validate_face(face_region)
                print(f"   Валідність: {'✅ Валідне' if is_valid else '❌ Невалідне'}")
        
        print(f"\n[4/4] 🎨 Намалювання прямокутників...")
        result = detector.draw_face_rectangles(processed, faces, color=(0, 255, 0), thickness=2)
        cv2.imwrite("output_faces_detected.jpg", result)
        print(f"✅ Результат збережено: output_faces_detected.jpg")
        
        print("\n" + "=" * 60)
        print("✅ ТЕСТИ ЗАВЕРШЕНІ!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_face_detector()
