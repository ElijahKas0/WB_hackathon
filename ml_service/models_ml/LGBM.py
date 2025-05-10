import joblib
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import (
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from scipy.sparse import issparse

def predict_model_LGBM(df_train,df_test):

    # Удаление дубликатов
    df_train = df_train.drop_duplicates()
    df_test = df_test.drop_duplicates()

    y_train_np = df_train['target'].to_numpy()
    y_test_np = df_test['target'].to_numpy()

    df_train.drop(columns=['target'], inplace=True)
    df_test.drop(columns=['target'], inplace=True)

    # Кодирование категориальных признаков
    categorical_cols = ['PaymentType', 'service']

    for i,df in enumerate([df_train, df_test]):
        df.drop(columns=['user_id', 'CreatedDate', 'NmAge','number_of_ordered_items'], inplace=True)
        df['IsPaid'] = df['IsPaid'].map({False: 0, True: 1})
        

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )


    X_train_transformed = preprocessor.fit_transform(df_train)
    X_test_transformed = preprocessor.transform(df_test)

    # Преобразование X в numpy
    if issparse(X_train_transformed):
        X_test_transformed = X_test_transformed.toarray()

    model = joblib.load('LGBM_model.pkl')

    # Предсказание вероятностей на тесте
    y_proba_test = model.predict_proba(X_test_transformed)[:, 1]

    # Подбор лучшего порога
    thresholds = np.arange(0.75, 0.961, 0.01)
    best_threshold = 0.75
    best_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

    for threshold in thresholds:
        y_pred_test = (y_proba_test >= threshold).astype(int)
        precision = precision_score(y_test_np, y_pred_test, zero_division=0)
        recall = recall_score(y_test_np, y_pred_test)

        if recall >= 0.06 and precision > best_metrics['precision']:
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1_score(y_test_np, y_pred_test)
            }

    # Финальные метрики
    y_pred_test = (y_proba_test >= best_threshold).astype(int)

    return y_pred_test, y_test_np 