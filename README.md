# Neural Chatbot (Seq2Seq + Attention)

Генеративный чат-бот на PyTorch. Модель учится на диалогах и сама генерирует ответы посимвольно.

## Особенности

- **Seq2Seq архитектура** с механизмом внимания (Attention)
- **Bidirectional LSTM** энкодер
- **7 оптимизаторов**: Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, NAdam
- **6 LR schedulers**: Step, Exponential, Cosine, Plateau, Warmup, Warmup+Cosine
- **10 готовых пресетов** конфигурации
- Полная настройка через `config.txt`
- Сохранение/загрузка модели

## Структура проекта

```
├── chatbot.py        # Основной скрипт
├── config.txt        # Конфигурация
├── presets.bat       # Пресеты конфигурации
├── run_chatbot.bat   # Запуск
└── data/
    ├── dialogs.txt   # Диалоги для обучения
    ├── model.pth     # Обученная модель
    └── tokenizer.json # Словарь
```

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/your-username/neural-chatbot.git
cd neural-chatbot

# Установить зависимости
pip install torch numpy
```

## Быстрый старт

1. Добавьте диалоги в `data/dialogs.txt`:
```
Q: привет
A: Привет! Как дела?

Q: как дела
A: Отлично, спасибо!
```

2. Запустите обучение:
```bash
python chatbot.py
```

Или используйте `run_chatbot.bat` на Windows.

## Конфигурация

Все настройки в файле `config.txt`:

### Модель
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| embed_dim | Размер эмбеддинга | 128 |
| hidden_dim | Размер скрытого слоя | 256 |
| max_len | Макс. длина текста | 128 |

### Обучение
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| epochs | Количество эпох | 150 |
| learning_rate | Скорость обучения | 0.002 |
| batch_size | Размер батча | 8 |
| optimizer | Оптимизатор | adam |
| lr_scheduler | Планировщик LR | none |

### Генерация
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| temperature | Креативность (0.1-2.0) | 0.7 |
| max_response_len | Макс. длина ответа | 150 |

## Оптимизаторы

| Название | Описание |
|----------|----------|
| `adam` | Стандартный Adam |
| `adamw` | Adam с weight decay |
| `sgd` | SGD с momentum |
| `rmsprop` | RMSprop |
| `adagrad` | Adagrad |
| `adadelta` | Adadelta |
| `nadam` | NAdam (Nesterov + Adam) |

## LR Schedulers

| Название | Описание |
|----------|----------|
| `none` | Постоянный LR |
| `step` | Уменьшение каждые N эпох |
| `exponential` | Экспоненциальное затухание |
| `cosine` | Косинусное затухание |
| `plateau` | Уменьшение при стагнации loss |
| `warmup` | Плавный разогрев |
| `warmup_cosine` | Разогрев + косинус |

## Пресеты

Запустите `presets.bat` для выбора готовой конфигурации:

| # | Пресет | Описание |
|---|--------|----------|
| 1 | Fast | Быстрое обучение (50 эпох) |
| 2 | Balanced | Сбалансированный (рекомендуется) |
| 3 | Quality | Качественное обучение (300 эпох) |
| 4 | Large | Большая модель (нужен GPU) |
| 5 | Small | Для слабых ПК |
| 6 | Creative | Высокая температура |
| 7 | Precise | Низкая температура |
| 8 | SGD | Классический SGD |
| 9 | AdamW | AdamW + Cosine |
| 10 | Experimental | NAdam + Warmup |

## Команды в чате

| Команда | Описание |
|---------|----------|
| `выход` / `quit` | Выйти |
| `debug` | Отладка последнего ответа |
| `temp X` | Установить температуру |
| `config` | Показать конфигурацию |
| `reload` | Перезагрузить конфиг |
| `files` | Показать файлы данных |

## Формат диалогов

```
Q: вопрос пользователя
A: ответ бота

Q: другой вопрос
A: другой ответ
```

- Каждый диалог = пара Q/A
- Пустые строки игнорируются
- Строки с `#` — комментарии
- Чем больше диалогов, тем лучше

## Требования

- Python 3.8+
- PyTorch 1.9+
- NumPy

## GPU

Модель автоматически использует CUDA если доступна:
```
[INFO] Устройство: cuda
[INFO] GPU: NVIDIA GeForce RTX 3080
```

Для CPU обучения уменьшите размер модели (пресет Small).

## Примеры конфигураций

### Быстрый тест
```
epochs = 50
embed_dim = 64
hidden_dim = 128
optimizer = adam
lr_scheduler = none
```

### Качественное обучение
```
epochs = 300
embed_dim = 256
hidden_dim = 512
optimizer = adamw
lr_scheduler = warmup_cosine
warmup_epochs = 30
```

### Для слабого ПК
```
epochs = 100
embed_dim = 32
hidden_dim = 64
batch_size = 16
optimizer = adam
```

## Лицензия

MIT License
