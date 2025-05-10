# backend/utils/validate_csv.py
import pandas as pd
from fastapi import UploadFile, HTTPException

def validate_csv(file: UploadFile) -> pd.DataFrame:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать CSV")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Файл пустой")
    
    if df.shape[1] < 1:
        raise HTTPException(status_code=400, detail="CSV не содержит таблицу")
    
    return df
