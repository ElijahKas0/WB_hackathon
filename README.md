# WB_hackathon

## 1. Руководство по использованию репозитория

Структура репозитория организована следующим образом:
├── ml_service/ # Директория с сервисом машинного обучения
│ ├── models_ml/ #Модели машинного обучения 
│ ├── static/
│ ├── templates/ 
│ ├── utils/ #Валидация csv
│ ├── routes.py
│ ├── main.py
│ ├── Dockerfile
│ ├── requirements.txt
│ └── docker-compose.yml # Конфигурация Docker
│
└── Top_3_notebooks/ # Лучшие модели (Jupyter Notebooks)
  ├── Top_1_notebook.ipynb # Ансамблевое решение Catboost и LightGBM 
  ├── Top_2_notebook.ipynb # Ансамблевое решение LightGBM и Logreg
  ├── Top_2_notebook.ipynb # Ансамблевое решение Catboost и Logreg
  │
  └── Task/ # Исходные данные
    ├── train.csv # Обучающая выборка
    └── test.csv # Тестовая выборка
## 2. Руководство по запуску сервиса

Сервис контейнеризован с помощью Docker. Для запуска:

1. Перейдите в директорию `ml_service`:
   ```pwsh
   cd ml_service
2. В командной строке введите команду:
   ```pwsh
   docker-compose up --build
3. После сборки API будет доступно::
   http://localhost:8005/docs
