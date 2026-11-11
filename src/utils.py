"""
Evaluation metrics and helper functions.
"""

import numpy as np
import polars as pl

def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """
    Jane Street yarışmasının resmi Ağırlıklı Sıfır-Ortalamalı R^2 (Weighted R2) metriği.
    """
    numerator = np.sum(weights * ((y_true - y_pred) ** 2))
    denominator = np.sum(weights * (y_true ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - (numerator / denominator))

def load_fold_arrays(parquet_path: str, feature_cols: list, target_col: str = "responder_6", max_rows: int = 3_000_000):
    """
    Colab RAM sınırını aşmamak için işlenmiş Parquet dosyasından veriyi 
    Float32 NumPy dizileri olarak belleğe çeker.
    """
    lazy_df = pl.scan_parquet(parquet_path)
    if max_rows is not None:
        lazy_df = lazy_df.tail(max_rows)
    
    df = lazy_df.collect(streaming=True)
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df[target_col].to_numpy().astype(np.float32)
    w = df["weight"].to_numpy().astype(np.float32)
    return X, y, w