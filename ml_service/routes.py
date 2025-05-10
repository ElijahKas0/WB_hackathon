import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils.validate_csv import validate_csv
from models_ml.LGBM import predict_model_LGBM
from models_ml.Catboost import predict_model_Catboost

router = APIRouter()
templates = Jinja2Templates(directory="templates")
executor = ThreadPoolExecutor()  # Используем глобально

# Обёртка для запуска функций в отдельном потоке
async def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    try:
        # Валидация и чтение файлов
        df_train = validate_csv(train_file)
        df_test = validate_csv(test_file)

        # Параллельная обработка
        task1 = run_in_threadpool(predict_model_LGBM, df_train.copy(), df_test.copy())
        task2 = run_in_threadpool(predict_model_Catboost, df_train.copy(), df_test.copy())

        (task1_result,  y_pred_test_Catboost) = await asyncio.gather(task1, task2)
        y_pred_test_LGBM, y_test_np = task1_result
        
        TP_lgbm_indices = np.where((y_pred_test_LGBM == 1) & (y_test_np == 1))[0]
        TP_catboost_indices = np.where((y_pred_test_Catboost == 1) & (y_test_np == 1))[0]

        # Все уникальные TP
        all_unique_TP = np.union1d(TP_lgbm_indices, TP_catboost_indices)
        
        # True Negatives - TN
        FP_lgbm_indices = np.where((y_pred_test_LGBM == 1) & (y_test_np == 0))[0]
        FP_catboost_indices = np.where((y_pred_test_Catboost == 1) & (y_test_np == 0))[0]

        # Все уникальные TN
        all_unique_FP = np.union1d(FP_lgbm_indices, FP_catboost_indices)

        target_1_test = np.sum(y_test_np)  
        target_0_test = len(y_test_np) - target_1_test  
        # Итоговая точность и полнота
        precision = len(all_unique_TP)/(len(all_unique_TP)+len(all_unique_FP))
        recall = len(all_unique_TP)/target_1_test
        
        TP = len(all_unique_TP)
        FP = len(all_unique_FP)
        FN = target_1_test - FP
        TN = target_0_test - TP
        conf_matrix = [
            [TN, FP],
            [FN, TP] 
        ]

        result_df = pd.DataFrame(
            conf_matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        
        result_html = result_df.to_html(classes="table table-bordered text-center align-middle")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "metric1": precision,
            "metric2": recall,
            "result": result_html
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })

