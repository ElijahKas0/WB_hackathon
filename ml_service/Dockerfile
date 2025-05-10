# Базовый образ Python
FROM python:3.10-slim

# Установка зависимостей системы (опционально, если pandas/ML-библиотекам нужно)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем всё в контейнер
COPY . .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Запуск приложения через Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
