# Основные библиотеки
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score
)
from sklearn.model_selection import (
    train_test_split
)

def predict_model_Catboost(df_train,df_test):

    # Удаление дубликатов
    df_raw_train = df_train.drop_duplicates()
    df_raw_test = df_test.drop_duplicates()

    # Кодирование категориальных признаков
    for df in [df_raw_train, df_raw_test]:
        df['IsPaid'] = df['IsPaid'].map({False: 0, True: 1})
        df['service'] = df['service'].map({'ordo': 1, 'nnsz': 2})
        df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])

    df_raw_train = pd.get_dummies(df_raw_train, columns=['PaymentType'], prefix='PaymentType', drop_first=True)
    df_raw_test = pd.get_dummies(df_raw_test, columns=['PaymentType'], prefix='PaymentType', drop_first=True)

    # Обработка Distance: обрезка выбросов и заполнение пропусков
    lower_quantile_dist = df_raw_train['Distance'].quantile(0.005)
    upper_quantile_dist = df_raw_train['Distance'].quantile(0.995)
    df_raw_train['Distance'] = df_raw_train['Distance'].clip(lower_quantile_dist, upper_quantile_dist).fillna(df_raw_train['Distance'].median())
    df_raw_test['Distance'] = df_raw_test['Distance'].clip(lower_quantile_dist, upper_quantile_dist).fillna(df_raw_test['Distance'].median())

    train_data = df_raw_train.drop(columns=['user_id', 'nm_id', 'CreatedDate'])
    test_data = df_raw_test.drop(columns=['user_id', 'nm_id', 'CreatedDate'])

    # Подготовка данных
    def prepare_data(df):
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df

    train_data = prepare_data(train_data)
    test_data = prepare_data(test_data)

    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])

    # Выравнивание признаков
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0
    X_test = X_test[X_train.columns]

    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    final_model = joblib.load('models_ml/catboost_model.pkl')

    # Подбор порога
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.5, 0.96, 10)
    best_threshold = 0.5
    best_metrics = {'precision': 0, 'recall': 0}

    for thresh in thresholds:
        y_val_pred = (y_val_proba >= thresh).astype(int)
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred)

        if precision >= 0.8 and recall >= 0.1 and precision > best_metrics['precision']:
            best_metrics = {'precision': precision, 'recall': recall}
            best_threshold = thresh

    # Оценка на тестовых данных
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    return y_test_pred