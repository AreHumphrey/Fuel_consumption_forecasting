import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from pathlib import Path
from src.config import CATBOOST_PARAMS, TARGET, FEATURES_NUMERIC, FEATURES_CATEGORICAL, MODEL_DIR

logger = logging.getLogger(__name__)


class FuelConsumptionModel:
    
    def __init__(self, params: dict = None):
        self.params = params or CATBOOST_PARAMS.copy()
        self.model = None
        self.feature_names = FEATURES_NUMERIC + FEATURES_CATEGORICAL
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, save_path: str = None) -> dict:
        train_pool = Pool(data=X_train, label=y_train, cat_features=FEATURES_CATEGORICAL, feature_names=self.feature_names)
        
        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(data=X_val, label=y_val, cat_features=FEATURES_CATEGORICAL, feature_names=self.feature_names)
        
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True, plot=False)
        
        metrics = {}
        if eval_pool is not None:
            y_pred = self.model.predict(eval_pool)
            metrics = {
                "mae": mean_absolute_error(y_val, y_pred), 
                "rmse": np.sqrt(mean_squared_error(y_val, y_pred)), 
                "r2": r2_score(y_val, y_pred)
            }
            logger.info(metrics)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(save_path)
            logger.info(save_path)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите train() сначала.")
        predict_pool = Pool(data=X, cat_features=FEATURES_CATEGORICAL, feature_names=self.feature_names)
        return self.model.predict(predict_pool)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        importance = self.model.get_feature_importance()
        df = pd.DataFrame({"feature": self.feature_names, "importance": importance}).sort_values("importance", ascending=False)
        return df.head(top_n)
    
    @classmethod
    def load(cls, model_path: str) -> "FuelConsumptionModel":
        instance = cls()
        instance.model = CatBoostRegressor()
        instance.model.load_model(model_path)
        return instance


def run_training(data_path: str, output_model: str):
    df = pd.read_csv(data_path)
    from src.features import prepare_features
    df = prepare_features(df)
    
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y = df[TARGET]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = FuelConsumptionModel()
    metrics = model.train(X_train, y_train, X_val, y_val, save_path=output_model)
    
    importance = model.get_feature_importance()
    logger.info(importance.to_string())
    
    return model, metrics