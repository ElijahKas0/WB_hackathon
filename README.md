# WB_hackathon

## 1. Руководство по использованию репозитория

### Структура репозитория организована следующим образом:

```
├── ml_service/
│   ├── models_ml/       # Модели ML
│   ├── static/          # CSS/JS
│   ├── templates/       # HTML шаблоны
│   ├── utils/           # Валидация CSV
│   ├── routes.py        # Fastapi routes
│   ├── main.py          # Главный скрипт
│   ├── Dockerfile       # Docker config
│   └── docker-compose.yml
│
└── Top_3_notebooks/     # Лучшие модели
   ├── Top_1.ipynb      # CatBoost + LightGBM
   ├── Top_2.ipynb      # LightGBM + LogReg
   ├── Top_3.ipynb      # CatBoost + LogReg
   └── Task/            # Данные
     ├── train.csv       # Обучающая
     └── test.csv         # Тестовая
```
    
## 2. Руководство по запуску сервиса

Сервис контейнеризован с помощью Docker. Для запуска:

1. Перейдите в директорию `ml_service`:
   ```pwsh
   cd ml_service
2. В командной строке введите команду:
   ```pwsh
   docker-compose up --build
3. После сборки API будет доступно::
   http://localhost:8005
