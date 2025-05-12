# Образ от которого будем наследоваться
FROM python:3.10-slim

# ставим, что нужно в контейнер и добавляем рабочую директорию 
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем всё в созданный контейнер и ставим питон-зависимости
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# запуск приложения и настройка параметров
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
