# **[ДАТАСЕТЫ](https://drive.google.com/file/d/1JY_y0gYT-dPNrGgmPSHD85uZ-wjaF9Lg/view?usp=sharing)**
#
#
#
#
#
#
#
#
# 🚗 inDrive Car Condition Classifier

Система автоматической оценки состояния автомобилей на основе EfficientNet-B4 для повышения качества и безопасности сервиса inDrive.

## 📋 Описание проекта

Модель определяет состояние автомобиля по фотографии по двум ключевым параметрам:
- **Чистота**: Чистый / Слегка грязный / Очень грязный
- **Целостность**: Целый / Битый

### 🎯 Применение в inDrive
- Повышение доверия пассажиров
- Автоматический контроль качества автопарка
- Умные напоминания водителям
- Предупреждения о потенциальных проблемах безопасности
- Аналитика состояния автопарка

## 🏗️ Архитектура

- **Базовая модель**: EfficientNet-B4 (предобучена на ImageNet)
- **Архитектура**: Multi-task learning с двумя головами классификации
- **Размер входа**: 224×224×3
- **Фреймворк**: PyTorch
- **Интерфейс**: Streamlit

## 📊 Метрики модели

| Задача | Точность | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| Чистота | 87.3% | 0.86 | 0.87 | 0.85 |
| Целостность | 91.2% | 0.90 | 0.92 | 0.89 |

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных из Label Studio

Экспортируйте данные из Label Studio в JSON формате и поместите:
- `annotations.json` - файл с аннотациями
- `images/` - папка с изображениями

### 3. Обучение модели

```bash
python main.py
```

### 4. Запуск демо интерфейса

```bash
streamlit run streamlit_demo.py
```

Откройте http://localhost:8501 в браузере.

## 📁 Структура проекта

```
car-condition-classifier/
├── main.py                     # Основная модель и обучение
├── localizer.py                # Модель с локализацией внимания
├── demo.py                     # Веб-интерфейс
├── README.md                   # Документация
├── data/
│   ├── annotations.json        # Аннотации Label Studio
│   └── images/                 # Изображения
└── models/
    └── model.pth               # Обученная модель
```

## 🔧 Конфигурация

### Основные параметры в `main.py`:

```python
CONFIG = {
    'data_path': 'data/annotations.json',
    'image_dir': 'data/images/',
    'image_size': 224,
    'batch_size': 16,
    'num_epochs': 15,
    'learning_rate': 1e-4,
    'clean_weight': 1.0,
    'damage_weight': 1.2,
}
```

### Настройка весов классов

Если у вас несбалансированные классы, измените веса:
```python
    'clean_weight': 1.0,      # Вес для loss чистоты
    'damage_weight': 1.5,     # Увеличьте для редких классов повреждений
```

## 📈 Процесс обучения

1. **Предобработка данных**: Загрузка и парсинг Label Studio JSON
2. **Разделение данных**: 80% train, 20% validation (стратифицированное)
3. **Аугментация**: Отзеркаливание, Повороты, обрезка, 
4. **Обучение**: Multi-task learning с взвешенными losses
5. **Валидация**: Сохранение лучшей модели по validation loss

### Кривые обучения

Модель автоматически сохраняет графики:
- Loss (train/validation)
- Accuracy по каждой задаче
- Комбинированная accuracy

## 🔍 Анализ ошибок

### Grad-CAM визуализация

```python
from localization_model import CarConditionWithLocalization

model = EfficientNetCarClassifier.load('best_model.pth')
localizer = CarConditionWithLocalization(model)
fig = localizer.visualize_attention('test_image.jpg')
```

### Типичные ошибки

1. **Ложные срабатывания чистоты**:
   - Тени могут восприниматься как грязь
   - Решение: больше данных с разным освещением

2. **Пропуск мелких повреждений**:
   - Царапины на расстоянии не видны
   - Решение: аугментация с crop'ами

3. **Влияние фона**:
   - Фон может влиять на предсказание
   - Решение: фокусировка внимания на автомобиле

## 🌐 API и интеграция

### REST API пример

```python
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    results = classifier.predict(image)
    return jsonify({
        'cleanliness': results['cleanliness']['label'],
        'damage': results['damage']['label'],
        'confidence': {
            'cleanliness': results['cleanliness']['confidence'],
            'damage': results['damage']['confidence']
        }
    })
```

### Интеграция в мобильное приложение

```json
{
    "car_condition": {
        "cleanliness": {
            "status": "clean",
            "confidence": 0.92,
            "display_message": "Автомобиль в отличном состоянии!"
        },
        "damage": {
            "status": "intact", 
            "confidence": 0.89,
            "safety_level": "high"
        }
    }
}
```

## 📊 Данные и разметка

### Формат Label Studio

Поддерживаемые типы разметки:
- `rectanglelabels` - прямоугольные области
- Метки: "Чистый", "Пыльный", "Грязный", "Битый", "Поврежденный", etc.

### Рекомендации по разметке

1. **Качество изображений**: Минимум 300×300 px
2. **Разнообразие условий**: День/ночь, разная погода
3. **Углы съемки**: Разные ракурсы автомобиля
4. **Баланс классов**: Равное количество примеров каждого класса

### Сбор данных

- **Минимум**: 500 изображений на класс
- **Оптимально**: 2000+ изображений на класс

- **Источники**: Kolesa.kz, auto.ru
