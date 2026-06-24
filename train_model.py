import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Загальні константи для подальшої обробки
IMG_SIZE = 160 # Зменшений розмір для швидшого навчання та збереження деталей
BATCH_SIZE = 32 # Збільшений розмір батчу для стабільнішого градієнта та кращої роботи з класами
EPOCHS_PHASE1 = 25 # К-ть епох для першої фази навчання (заморожені шари) - збільшено для кращого навчання верхніх шарів
EPOCHS_PHASE2 = 15 # К-ть епох для другої фази (розморожування) - зменшено, щоб уникнути перенавчання

# Шляхи до даних та моделі
TRAIN_DIR = "data/rafdb/train" 
VAL_DIR = "data/rafdb/test"
MODEL_PATH = "emotion_model.h5"

# Змінні для зберігання класів та їх ваг
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input, # Використання вбудованої функції для MobileNetV2
    rotation_range = 10, # Невеликий діапазон для обертання, щоб зберегти розпізнаваність емоцій
    width_shift_range = 0.05, # Зсув по ширині для різноманітності позицій обличчя
    height_shift_range = 0.05, # Зсув по висоті для різноманітності позицій обличчя
    horizontal_flip = True, # Горизонтальне відображення для збільшення різноманітності (емоції не залежать від сторони обличчя)
    zoom_range = 0.1, # Зум для різноманітності розмірів обличчя
    brightness_range = [0.8, 1.2] # Зміна яскравості для різноманітності освітлення
)

# Валідаційний генератор без аугментації, лише з препроцесінгом
val_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)

# Генератори для тренування та валідації
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = True
)

# Валідаційний генератор без перемішування, щоб зберегти порядок для обчислення класових ваг
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = False
)

# Обчислення класових ваг для балансування класів
labels = train_generator.classes

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

# Створення словника класових ваг для передачі в модель
class_weights = dict(
    zip(np.unique(labels), class_weights_arr)
)

print("Class weights:", class_weights)

# Завантаження існуючої моделі або створення нової
if os.path.exists(MODEL_PATH):
    print("\nЗавантаження наявної моделі...")
    model = load_model(MODEL_PATH)
else:
    print("\n🆕 Створення нової моделі...")

    # Використання MobileNetV2 як базової моделі з попередньо навченими вагами на ImageNet
    base_model = MobileNetV2(
        weights = 'imagenet',
        include_top = False,
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
    )

    # Заморожування базової моделі для першої фази навчання
    for layer in base_model.layers:
        layer.trainable = False

    # Додавання власних класифікаційних шарів поверх базової моделі
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation = 'relu')(x)
    x = Dropout(0.4)(x)

    # Вихідний шар з 7 класами та softmax активацією для багатокласової класифікації
    output = Dense(7, activation = 'softmax')(x)

    model = Model(inputs = base_model.input, outputs = output)

# Компіляція моделі з оптимізатором Adam
model.compile(
    optimizer = Adam(learning_rate=1e-4),
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics = ['accuracy']
)

# Виведення архітектури моделі
model.summary()

callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3, verbose=1)
]

# Тренування моделі з замороженими шарами базової моделі (етап 1)
print("\nЕтап 1: Навчання з замороженими шарами базової моделі...")

# Навчання моделі з використанням генераторів та класових ваг для балансування класів
model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = EPOCHS_PHASE1,
    class_weight = class_weights,
    callbacks = callbacks
)

# Fine-tuning. Розморожування верхніх шарів базової моделі (етап 2)
print("\nЕтап 2: Fine-tuning...")

# Верхні шари
for layer in model.layers[:-40]:
    layer.trainable = False

# Розморожування шарів
for layer in model.layers[-40:]:
    layer.trainable = True

# Компіляція моделі з меншим learning rate для fine-tuning
model.compile(
    optimizer = Adam(learning_rate=1e-5),
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics = ['accuracy']
)

# Навчання моделі з розмороженими шарами базової моделі
model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = EPOCHS_PHASE2,
    class_weight = class_weights,
    callbacks = callbacks
)

# Збереження моделі після навчання
model.save(MODEL_PATH)
print("\nМодель успішно збережена за шляхом:", MODEL_PATH)