"""
Baseline and proposed model implementations for Jane Street Market Forecasting.
"""
import numpy as np
from sklearn.linear_model import Ridge
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

class ZeroBaseline:
    """Tüm gözlemlere 0.0 atayan resmi naif referans baseline modeli."""
    def fit(self, X_train, y_train=None, w_train=None, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=np.float32)


class RidgeBaseline:
    """Çoklu korelasyona karşı L2 regülarizasyonlu doğrusal regresyon baseline modeli."""
    def __init__(self, alpha: float = 100.0, random_state: int = 42):
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X_train, y_train, w_train=None, **kwargs):
        self.model.fit(X_train, y_train, sample_weight=w_train)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, -5.0, 5.0).astype(np.float32)


class LGBMBaseline:
    """Hızlı yaprak odaklı (leaf-wise) LightGBM baseline modeli."""
    def __init__(self, params: dict = None):
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X_train, y_train, w_train=None, X_val=None, y_val=None, w_val=None):
        self.model = lgb.LGBMRegressor(**self.params)
        fit_kwargs = {"sample_weight": w_train}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_sample_weight"] = [w_val]
            fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, -5.0, 5.0).astype(np.float32)


class XGBoostBaseline:
    """Histogram tabanlı derinlik odaklı (depth-wise) XGBoost baseline modeli."""
    def __init__(self, params: dict = None):
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X_train, y_train, w_train=None, X_val=None, y_val=None, w_val=None):
        fit_params = self.params.copy()
        if X_val is None or y_val is None:
            fit_params.pop("early_stopping_rounds", None)
            
        self.model = XGBRegressor(**fit_params)
        fit_kwargs = {"sample_weight": w_train, "verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if w_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [w_val]
                
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, -5.0, 5.0).astype(np.float32)


class CatBoostBaseline:
    """Gürültüye dayanıklı simetrik ağaçlı CatBoost baseline modeli."""
    def __init__(self, params: dict = None):
        default_params = {
            "iterations": 400,
            "learning_rate": 0.06,
            "depth": 6,
            "loss_function": "RMSE",
            "random_seed": 42,
            "thread_count": -1,
            "verbose": 0
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X_train, y_train, w_train=None, X_val=None, y_val=None, w_val=None):
        self.model = CatBoostRegressor(**self.params)
        fit_kwargs = {"sample_weight": w_train, "verbose": False}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = 30
            
        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.clip(preds, -5.0, 5.0).astype(np.float32)