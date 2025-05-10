# Основные библиотеки
import numpy as np
import pandas as pd
import warnings
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,precision_score, recall_score
)
from sklearn.preprocessing import (
    KBinsDiscretizer, StandardScaler,
)
from sklearn.model_selection import (
    train_test_split
)
warnings.filterwarnings('ignore')

def predict_model_Catboost(df_train,df_test):

    # Удаление дубликатов
    df_raw_train = df_train.drop_duplicates()
    df_raw_test = df_test.drop_duplicates()

    # Проверка начального числа строк
    print(f"Количество строк в df_raw_train до обработки: {len(df_raw_train)}")
    print(f"Количество строк в df_raw_test до обработки: {len(df_raw_test)}")

    # Проверка дубликатов
    print(f"Количество дубликатов в df_raw_train: {df_raw_train.duplicated().sum()}")
    print(f"Количество дубликатов в df_raw_test: {df_raw_test.duplicated().sum()}")

    # Кодирование категориальных признаков
    for df in [df_raw_train, df_raw_test]:
        df['IsPaid'] = df['IsPaid'].map({False: 0, True: 1})
        df['service'] = df['service'].map({'ordo': 1, 'nnsz': 2})
        df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])

    df_raw_train = pd.get_dummies(df_raw_train, columns=['PaymentType'], prefix='PaymentType', drop_first=True)
    df_raw_test = pd.get_dummies(df_raw_test, columns=['PaymentType'], prefix='PaymentType', drop_first=True)

    train = pd.DataFrame()
    test = pd.DataFrame()

    # Обработка Distance: обрезка выбросов и заполнение пропусков
    lower_quantile_dist = df_raw_train['Distance'].quantile(0.005)
    upper_quantile_dist = df_raw_train['Distance'].quantile(0.995)
    df_raw_train['Distance'] = df_raw_train['Distance'].clip(lower_quantile_dist, upper_quantile_dist).fillna(df_raw_train['Distance'].median())
    df_raw_test['Distance'] = df_raw_test['Distance'].clip(lower_quantile_dist, upper_quantile_dist).fillna(df_raw_test['Distance'].median())

    # Feature engineering
    train['log_distance'] = np.log1p(df_raw_train['Distance'])
    test['log_distance'] = np.log1p(df_raw_test['Distance'])

    # Признак: mean_percent_of_ordered_items
    lower_quantile_percent = df_raw_train['mean_percent_of_ordered_items'].quantile(0.005)
    upper_quantile_percent = df_raw_train['mean_percent_of_ordered_items'].quantile(0.995)
    train['mean_percent_of_ordered_items'] = np.log1p(df_raw_train['mean_percent_of_ordered_items'].clip(lower_quantile_percent, upper_quantile_percent))
    lower_quantile_percent_test = df_raw_test['mean_percent_of_ordered_items'].quantile(0.005)
    upper_quantile_percent_test = df_raw_test['mean_percent_of_ordered_items'].quantile(0.995)
    test['mean_percent_of_ordered_items'] = np.log1p(df_raw_test['mean_percent_of_ordered_items'].clip(lower_quantile_percent_test, upper_quantile_percent_test))

    # Нормализация и бининг
    scaler_percent = StandardScaler()
    train['mean_percent_of_ordered_items'] = scaler_percent.fit_transform(train[['mean_percent_of_ordered_items']])
    test['mean_percent_of_ordered_items'] = scaler_percent.transform(test[['mean_percent_of_ordered_items']])
    discretizer_percent = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    train['mean_percent_of_ordered_items'] = discretizer_percent.fit_transform(train[['mean_percent_of_ordered_items']])
    test['mean_percent_of_ordered_items'] = discretizer_percent.transform(test[['mean_percent_of_ordered_items']])

    # Признак: is_new_account
    train['is_new_account'] = (df_raw_train['DaysAfterRegistration'] < 1000).astype(int)
    test['is_new_account'] = (df_raw_test['DaysAfterRegistration'] < 1000).astype(int)

    # Добавление целевой переменной
    train['target'] = df_raw_train['target']
    test['target'] = df_raw_test['target']

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
    y_test = test_data['target']

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

    # Лучшие параметры
    best_params = {'iterations': 800,
                'depth': 9,
                'learning_rate': 0.1,
                'l2_leaf_reg': 50,
                'scale_pos_weight': 9,
                'min_data_in_leaf': 50,
                'max_ctr_complexity': 3}

    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    # Финальная модель
    final_model = CatBoostClassifier(
        **best_params,
        eval_metric='F1',
        verbose=100,
        allow_writing_files=False
    )

    final_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val]),
        plot=False
    )

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