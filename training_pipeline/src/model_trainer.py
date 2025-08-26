import pandas as pd
import logging
import re
from typing import Optional, Tuple, Dict
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
from datetime import datetime
from comet_ml import Experiment

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train Prophet and XGBoost models for a given SKU."""
    
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
    
    def train_prophet(self, df: pd.DataFrame) -> Tuple[Optional[Prophet], Optional[Dict[str, float]]]:
        """Train a Prophet model."""
        try:
            prophet_df = df[['OrderDate', 'NoOfPItems']].rename(columns={'OrderDate': 'ds', 'NoOfPItems': 'y'})
            if len(prophet_df) < 2:
                logger.warning("Insufficient data for Prophet training")
                return None, None
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                n_changepoints=min(25, len(prophet_df) // 2)
            )
            
            train, test = train_test_split(prophet_df, test_size=0.2, shuffle=False)
            if len(test) < 1:
                logger.warning("No test data for Prophet evaluation")
                return None, None
            
            model.fit(train)
            future = model.make_future_dataframe(periods=len(test), freq='ME')
            forecast = model.predict(future)
            y_pred = forecast[-len(test):]['yhat']
            y_true = test['y']
            mae = mean_absolute_error(y_true, y_pred)
            
            return model, {'mae': mae}
        
        except Exception as e:
            logger.error(f"Error training Prophet: {str(e)}")
            return None, None
    
    def train_xgboost(self, df: pd.DataFrame) -> Tuple[Optional[XGBRegressor], Optional[Dict[str, float]]]:
        """Train an XGBoost model."""
        try:
            X = df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
            y = df['NoOfPItems']
            if len(X) < 2:
                logger.warning("Insufficient data for XGBoost training")
                return None, None
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            if len(X_test) < 1:
                logger.warning("No test data for XGBoost evaluation")
                return None, None
            
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            return model, {'mae': mae}
        
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            return None, None
    
    def save_model(self, model, model_type: str, sku: str, model_dir: str) -> None:
        """Save model to disk and log as experiment asset."""
        try:
            safe_sku = re.sub(r'[^\w\-]', '_', sku.strip()).replace('__', '_')  # Avoid double underscores
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = Path(model_dir) / f"{safe_sku}_{model_type}_{timestamp}.pkl"
            joblib.dump(model, model_path)
            self.experiment.log_asset(str(model_path), file_name=f"{safe_sku}_{model_type}_{timestamp}.pkl")
            logger.info(f"Saved and logged model {model_type} for SKU {sku} at {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_type} for SKU {sku}: {str(e)}")
            raise