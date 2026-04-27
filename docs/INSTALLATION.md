# Встановлення та запуск

## 🔧 Вимоги

* Python 3.8+
* pip

---

## 📥 Клонування репозиторію

```bash
git clone https://github.com/klymsk/UniversityProject.git
cd emotion_recognition
```

---

## 📦 Встановлення залежностей

```bash
pip install -r requirements.txt
```

---

## ▶️ Запуск програми

```bash
python main.py
```

---

## 🧪 Запуск тестів

```bash
pytest tests/
```

---

## ⚠️ Можливі проблеми

### ❌ Не знаходиться модель

Переконайтесь, що файл моделі знаходиться у папці:

```
models/
```

---

### ❌ Помилки з OpenCV

Спробуйте:

```bash
pip install opencv-python
```

---
