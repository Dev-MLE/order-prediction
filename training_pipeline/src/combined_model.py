import pandas as pd
import numpy as np
from comet_ml import API, Experiment
from sklearn.ensemble import VotingRegressor
from sklearn.base import BaseEstimator
from pathlib import Path
import joblib
import logging
import os
from dotenv import load_dotenv
import re
from typing import Dict, Optional
from prophet import Prophet
from xgboost import XGBRegressor

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelCombiner:
    """Combine all trained Prophet and XGBoost models from a Comet.ml experiment into one final model."""
    
    def __init__(self, api_key: str, workspace: str, project_name: str, features_path: str, good_skus_path: str, experiment_id: Optional[str] = None):
        self.api = API(api_key=api_key)
        self.workspace = workspace
        self.project_name = project_name
        self.features_path = features_path
        self.good_skus_path = good_skus_path
        self.experiment_id = experiment_id
        self.df = None
        self.good_skus = None
        self.models = {}
        self.performance = {}
        self.load_data()
    
    def load_data(self) -> None:
        """Load features and good SKUs."""
        try:
            self.df = pd.read_parquet(self.features_path)
            self.df['SkuName'] = self.df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            logger.info(f"Loaded features dataset with {len(self.df)} rows and {len(self.df['SkuName'].unique())} unique SKUs")
            
            self.good_skus = pd.read_csv(self.good_skus_path)['SkuName'].unique().tolist()
            logger.info(f"Loaded {len(self.good_skus)} good SKUs")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def fetch_models_and_performance(self) -> None:
        """Fetch models from the specified or first experiment's artifacts and performance metrics."""
        try:
            # Get experiments
            experiments = self.api.get_experiments(workspace=self.workspace, project_name=self.project_name)
            if not experiments:
                logger.error("No experiments found in project")
                raise ValueError("No experiments found")
            
            # Select experiment
            if self.experiment_id:
                latest_experiment = next((exp for exp in experiments if exp.id == self.experiment_id), None)
                if not latest_experiment:
                    logger.error(f"Experiment {self.experiment_id} not found")
                    raise ValueError(f"Experiment {self.experiment_id} not found")
            else:
                latest_experiment = experiments[0]
                logger.warning("No experiment ID specified, using first experiment available")
            
            logger.info(f"Using experiment: {latest_experiment.id}")
            
            # Fetch performance metrics
            sku_map = {}
            params = latest_experiment.get_parameters_summary()
            for param in params:
                if param['name'].startswith('cleaned_sku_') or param['name'].startswith('outliers_sku_'):
                    try:
                        step = int(param['name'].split('_')[-1])
                        dataset_type = 'cleaned' if param['name'].startswith('cleaned') else 'outliers'
                        value = param.get('valueCurrent', f"Missing_Value_{step}")
                        if isinstance(value, str):
                            value = re.sub(r'\s+', ' ', value.strip())
                        sku_map[(latest_experiment.id, step, dataset_type)] = value
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid parameter {param['name']} in experiment {latest_experiment.id}: {str(e)}")
            
            metrics = latest_experiment.get_metrics()
            for metric in metrics:
                if metric['metricName'] in ['cleaned_ensemble_mae', 'outliers_ensemble_mae']:
                    if metric['step'] is None:
                        logger.warning(f"Skipping metric {metric['metricName']} with None step in experiment {latest_experiment.id}")
                        continue
                    try:
                        step = int(metric['step'])
                        dataset_type = 'cleaned' if metric['metricName'].startswith('cleaned') else 'outliers'
                        sku = sku_map.get((latest_experiment.id, step, dataset_type), f"Unknown_SKU_{step}")
                        mae = float(metric['metricValue'])
                        self.performance[f"{dataset_type}_{sku}"] = mae
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping metric {metric['metricName']} in experiment {latest_experiment.id}: {str(e)}")
            
            logger.info(f"Fetched performance for {len(self.performance)} SKUs")
            
            # Fetch model artifacts from the experiment
            artifacts = latest_experiment.get_asset_list()
            model_artifacts = [a for a in artifacts if a['fileName'].endswith('.pkl') and ('_prophet' in a['fileName'] or '_xgboost' in a['fileName'])]
            for artifact in model_artifacts:
                model_name = artifact['fileName'].replace('.pkl', '')
                parts = model_name.rsplit('_', 2)
                if len(parts) < 3:
                    logger.warning(f"Skipping invalid artifact name: {model_name}")
                    continue
                sku = parts[0]
                model_type = parts[1]
                dataset_type = 'cleaned' if 'cleaned' in model_name else 'outliers'
                temp_path = Path('/tmp') / f"{model_name}.pkl"
                try:
                    asset_data = latest_experiment.get_asset(asset_id=artifact['assetId'], return_type='binary')
                    with open(temp_path, 'wb') as f:
                        f.write(asset_data)
                    model = joblib.load(temp_path)
                    if isinstance(model, (Prophet, XGBRegressor)):
                        # Sanitize key to avoid double underscores
                        sanitized_key = f"{dataset_type}_{sku}_{model_type}".replace('__', '_')
                        self.models[sanitized_key] = model
                        logger.info(f"Loaded model {model_name} for SKU {sku} ({dataset_type}, {model_type})")
                    else:
                        logger.warning(f"Artifact {model_name} is not a Prophet or XGBoost model")
                except Exception as e:
                    logger.warning(f"Failed to load artifact {model_name}: {str(e)}")
            
            logger.info(f"Fetched {len(self.models)} models from experiment {latest_experiment.id}")
        
        except Exception as e:
            logger.error(f"Error fetching models and performance: {str(e)}")
            raise
    
    def combine_models(self) -> BaseEstimator:
        """Combine trained models into a single VotingRegressor."""
        try:
            if not self.models:
                logger.error("No models fetched, cannot combine")
                raise ValueError("No models fetched")
            
            # Calculate weights based on inverse MAE (lower MAE = higher weight)
            total_mae = sum(1 / mae for mae in self.performance.values() if mae > 0)
            weights = {
                f"{dataset_type}_{sku}": (1 / mae) / total_mae if mae > 0 else 0
                for (dataset_type, sku), mae in [(k.split('_', 2)[:2], v) for k, v in self.performance.items()]
            }
            
            # Create estimators for VotingRegressor (only XGBoost models)
            estimators = [
                (key.replace('__', '_'), model)  # Sanitize estimator name
                for key, model in self.models.items()
                if isinstance(model, XGBRegressor)
            ]
            
            if not estimators:
                logger.error("No XGBoost models available for VotingRegressor")
                raise ValueError("No XGBoost models available")
            
            # Assign weights to XGBoost models
            voting_weights = [
                weights.get(key.rsplit('_', 1)[0].replace('__', '_'), 1.0 / len(estimators))
                for key, _ in estimators
            ]
            
            voting_regressor = VotingRegressor(
                estimators=estimators,
                weights=voting_weights
            )
            
            # Train on all good SKUs
            good_df = self.df[self.df['SkuName'].isin(self.good_skus)]
            X = good_df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
            y = good_df['NoOfPItems']
            
            if len(X) < 1 or y.isna().sum() >= len(y):
                logger.error("Insufficient valid data for training final model")
                raise ValueError("Insufficient valid data")
            
            voting_regressor.fit(X, y)
            logger.info("Combined XGBoost models into final voting regressor")
            
            return voting_regressor
        
        except Exception as e:
            logger.error(f"Error combining models: {str(e)}")
            raise
    
    def upload_final_model(self, model: BaseEstimator) -> None:
        """Upload the final model to Comet.ml Model Registry and save locally."""
        try:
            # Save locally
            local_model_path = Path('/workspaces/Machine-learning/models/final_general_model.pkl')
            local_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, local_model_path)
            logger.info(f"Saved final general model locally to {local_model_path}")
            
            # Upload to Comet.ml
            model_path = Path('/tmp/final_general_model.pkl')
            joblib.dump(model, model_path)
            experiment = Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=self.project_name,
                workspace=self.workspace
            )
            experiment.set_name("final_general_model")
            experiment.log_model("final_general_model", str(model_path))
            experiment.register_model("final_general_model")
            experiment.end()
            logger.info("Uploaded and registered final general model to Comet.ml Model Registry")
        
        except Exception as e:
            logger.error(f"Error uploading final model: {str(e)}")
            raise