"""
Тестовий скрипт для модуля Emotion Classifier
Демонстрація побудови та тестування моделі VGG16
"""

import numpy as np
from modules.emotion_classifier import EmotionClassifier
import cv2


def test_emotion_classifier():
    """Тестування EmotionClassifier"""
    
    print("=" * 60)
    print("🧪 ТЕСТУВАННЯ EMOTION CLASSIFIER")
    print("=" * 60)
    
    try:
        # Крок 1: Ініціалізація
        print("\n[1/5] 🔧 Ініціалізація класифікатора...")
        classifier = EmotionClassifier()
        print(f"✅ Класи емоцій: {classifier.EMOTION_CLASSES}")
        print(f"✅ Кількість класів: {classifier.NUM_CLASSES}")
        print(f"✅ Розмір вхідних даних: {classifier.INPUT_SIZE}")
        
        # Крок 2: Побудова моделі
        print("\n[2/5] 🔨 Побудова моделі VGG16 з transfer learning...")
        model = classifier.build_transfer_learning_model()
        print("✅ Модель побудована успішно!")
        
        # Крок 3: Тест препроцесування
        print("\n[3/5] 🎨 Тест препроцесування зображення...")
        
        # Створюємо тестове зображення (48x48, як у FER2013)
        test_face = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        
        # Препроцесуємо
        processed = classifier.preprocess_face(test_face)
        print(f"✅ Вхідне зображення: {test_face.shape}")
        print(f"✅ Оброблене для моделі: {processed.shape}")
        print(f"✅ Діапазон значень: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Крок 4: Тест передбачення
        print("\n[4/5] 🎯 Тест передбачення емоції...")
        emotion, confidence = classifier.predict_emotion(test_face)
        print(f"✅ Передбачена емоція: {emotion}")
        print(f"✅ Впевненість: {confidence:.2f}%")
        
        # Крок 5: Тест ймовірностей всіх емоцій
        print("\n[5/5] 📊 Тест передбачення ймовірностей для всіх емоцій...")
        probabilities = classifier.predict_emotions_probabilities(test_face)
        print("✅ Ймовірності емоцій:")
        for emotion, prob in probabilities.items():
            bar = "█" * int(prob * 50)  # Візуальний bar chart
            print(f"   {emotion:12} {prob:6.2%} {bar}")
        
        print("\n" + "=" * 60)
        print("✅ ВСІ ТЕСТИ EMOTION CLASSIFIER ПРОЙШЛИ!")
        print("=" * 60)
        
        print("\n📝 ІНФОРМАЦІЯ ПРО МОДЕЛІ:")
        print(f"   - Тип: VGG16 з Transfer Learning")
        print(f"   - Базові шари: Заморожені (не навчаються)")
        print(f"   - Спеціалізовані шари: 512->256->128->7")
        print(f"   - Optimizer: Adam (lr=1e-4)")
        print(f"   - Loss: Categorical Crossentropy")
        print(f"   - Готова до навчання на FER2013!")
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_emotion_classifier()
