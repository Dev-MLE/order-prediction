import pandas as pd
import logging
from typing import Dict, Optional
from comet_ml import Experiment
from config import TrainingPipelineConfig
from model_trainer import ModelTrainer
from pathlib import Path

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Pipeline to train models for all SKUs."""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.experiment = Experiment(
            api_key=config.comet_api_key,
            project_name=config.project_name,
            workspace=config.workspace
        )
        self.experiment.set_name("training_pipeline")
        self.trainer = ModelTrainer(experiment=self.experiment)
    
    def run(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Run the training pipeline for all SKUs."""
        try:
            results = {}
            df = pd.read_parquet(self.config.features_path)
            df['SkuName'] = df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            good_skus = pd.read_csv(self.config.good_skus_path)['SkuName'].unique()
            logger.info(f"Loaded {len(good_skus)} good SKUs")
            
            for sku in good_skus:
                sku_data = df[df['SkuName'] == sku]
                if len(sku_data) < 2:
                    logger.warning(f"Skipping SKU {sku} with insufficient data ({len(sku_data)} rows)")
                    continue
                
                step = len(results) + 1
                
                # Train Prophet
                prophet_model, prophet_metrics = self.trainer.train_prophet(sku_data)
                if prophet_model and prophet_metrics:
                    self.trainer.save_model(prophet_model, 'prophet', sku, self.config.model_dir)
                    self.experiment.log_metrics(
                        {f"cleaned_ensemble_mae": prophet_metrics['mae']},
                        step=step
                    )
                    self.experiment.log_parameters(
                        {f"cleaned_sku_{step}": sku},
                        step=step
                    )
                    results[f"cleaned_{sku}"] = prophet_metrics
                
                # Train XGBoost
                xgboost_model, xgboost_metrics = self.trainer.train_xgboost(sku_data)
                if xgboost_model and xgboost_metrics:
                    self.trainer.save_model(xgboost_model, 'xgboost', sku, self.config.model_dir)
                    self.experiment.log_metrics(
                        {f"cleaned_ensemble_mae": xgboost_metrics['mae']},
                        step=step
                    )
                    self.experiment.log_parameters(
                        {f"cleaned_sku_{step}": sku},
                        step=step
                    )
                    results[f"cleaned_{sku}"] = xgboost_metrics
            
            logger.info(f"Training completed for {len(results)} SKUs")
            self.experiment.end()
            return results
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            self.experiment.log_other("pipeline_error", str(e))
            self.experiment.end()
            raise