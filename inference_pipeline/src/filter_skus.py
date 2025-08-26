import pandas as pd
import logging
from comet_ml import API
from typing import Tuple
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SKUFilter:
    def __init__(self, api_key: str, workspace: str, project_name: str, experiment_id: str, threshold_perc: float = 0.05, metric_name: str = 'cleaned_ensemble_mae'):
        """Initialize SKU filter with Comet.ml credentials and threshold."""
        self.api = API(api_key=api_key)
        self.workspace = workspace
        self.project_name = project_name
        self.experiment_id = experiment_id
        self.threshold_perc = threshold_perc
        self.metric_name = metric_name
        self.fallback_metric = 'cleaned_xgboost_mae'
    
    def fetch_metrics(self) -> pd.DataFrame:
        """Fetch specified MAE metrics from Comet.ml."""
        try:
            experiment = self.api.get_experiment(self.workspace, self.project_name, self.experiment_id)
            if not experiment:
                logger.error(f"No experiment found for ID {self.experiment_id}")
                raise ValueError("Experiment not found")
            
            metrics = experiment.get_metrics()
            logger.debug(f"Raw metrics response: {json.dumps(metrics, indent=2)}")
            metric_names = {metric['metricName'] for metric in metrics}
            logger.info(f"Available metrics in experiment: {sorted(metric_names)}")
            
            sku_metrics = []
            sku_map = {}
            params = experiment.get_parameters_summary()
            logger.debug(f"Raw parameters response: {json.dumps(params, indent=2)}")
            logger.info(f"Found {len(params)} parameters in experiment")
            
            for param in params:
                if param['name'].startswith('cleaned_sku_'):
                    try:
                        step = int(param['name'].split('_')[-1])
                        # Normalize SKU name using re.sub
                        sku_name = param.get('valueCurrent', f"Missing_Value_{step}").strip()
                        sku_name = re.sub(r'\s+', ' ', sku_name)
                        sku_map[step] = sku_name
                    except ValueError:
                        logger.warning(f"Invalid step format in parameter {param['name']}")
                        continue
            
            logger.debug(f"Mapped sku_map: {sku_map}")
            
            for metric in metrics:
                if metric['metricName'] == self.metric_name:
                    step = int(metric['step'])
                    sku = sku_map.get(step, f"Unknown_SKU_{step}")
                    mae = float(metric['metricValue'])
                    sku_metrics.append({'SkuName': sku, 'mae': mae, 'step': step})
            
            if not sku_metrics:
                logger.error("No cleaned_ensemble_mae metrics found")
                raise ValueError("No cleaned_ensemble_mae metrics found")
            
            metrics_df = pd.DataFrame(sku_metrics)
            logger.info(f"Fetched {len(metrics_df)} SKU metrics from Comet.ml")
            
            unknown_skus = metrics_df[metrics_df['SkuName'].str.startswith('Unknown_SKU_') | metrics_df['SkuName'].str.startswith('Missing_Value_')]
            if not unknown_skus.empty:
                logger.warning(f"Found {len(unknown_skus)} SKUs with no matching parameter: {unknown_skus['SkuName'].tolist()}")
            
            return metrics_df
        
        except Exception as e:
            logger.error(f"Error fetching metrics: {str(e)}")
            raise
    
    def filter_skus(self, features_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter SKUs based on performance."""
        try:
            df = pd.read_parquet(features_path)
            df['SkuName'] = df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            logger.info(f"Loaded original dataset with {len(df)} rows and {len(df['SkuName'].unique())} unique SKUs")
            
            avg_demand = df.groupby('SkuName')['NoOfPItems'].mean().reset_index()
            avg_demand.columns = ['SkuName', 'avg_demand']
            
            metrics_df = self.fetch_metrics()
            metrics_df['SkuName'] = metrics_df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            
            sku_metrics_df = metrics_df.merge(avg_demand, on='SkuName', how='left')
            sku_metrics_df['performance'] = sku_metrics_df['mae'] / sku_metrics_df['avg_demand'].replace(0, 1)
            
            unmatched_skus = sku_metrics_df[sku_metrics_df['avg_demand'].isna()]
            if not unmatched_skus.empty:
                logger.warning(f"Found {len(unmatched_skus)} SKUs with no demand data: {unmatched_skus['SkuName'].tolist()}")
            
            good_skus = sku_metrics_df[sku_metrics_df['performance'] < self.threshold_perc]
            improvement_skus = sku_metrics_df[sku_metrics_df['performance'] >= self.threshold_perc]
            
            # Deduplicate SKUs
            good_skus = good_skus[['SkuName', 'mae', 'step', 'avg_demand', 'performance']].drop_duplicates()
            improvement_skus = improvement_skus[['SkuName', 'mae', 'step', 'avg_demand', 'performance']].drop_duplicates()
            
            good_skus_data = good_skus.merge(df, on='SkuName', how='inner')
            improvement_skus_data = improvement_skus.merge(df, on='SkuName', how='inner')
            
            Path('/workspaces/Machine-learning/data/filtered').mkdir(parents=True, exist_ok=True)
            good_skus_data.to_csv('/workspaces/Machine-learning/data/filtered/good_skus.csv', index=False)
            improvement_skus_data.to_csv('/workspaces/Machine-learning/data/filtered/improvement_skus.csv', index=False)
            
            logger.info(f"Good SKUs: {len(good_skus)} unique SKUs (saved to good_skus.csv with {len(good_skus_data)} rows)")
            logger.info(f"Improvement SKUs: {len(improvement_skus)} unique SKUs (saved to improvement_skus.csv with {len(improvement_skus_data)} rows)")
            
            logger.info("\nSample Good SKUs (first 3):")
            for _, row in good_skus.head(3).iterrows():
                logger.info(f"SKU: {row['SkuName']}, MAE: {row['mae']:.4f}, Avg Demand: {row['avg_demand']:.4f}, Performance: {row['performance']:.4f}")
            
            logger.info("\nSample Improvement SKUs (first 3):")
            for _, row in improvement_skus.head(3).iterrows():
                logger.info(f"SKU: {row['SkuName']}, MAE: {row['mae']:.4f}, Avg Demand: {row['avg_demand']:.4f}, Performance: {row['performance']:.4f}")
            
            if not good_skus_data.empty:
                sample_sku = good_skus_data['SkuName'].iloc[0]
                logger.info(f"\nData Nature for Good SKU '{sample_sku}' (first 2 rows):")
                logger.info(good_skus_data[good_skus_data['SkuName'] == sample_sku].head(2).to_string(index=False))
            
            if not improvement_skus_data.empty:
                sample_sku = improvement_skus_data['SkuName'].iloc[0]
                logger.info(f"\nData Nature for Improvement SKU '{sample_sku}' (first 2 rows):")
                logger.info(improvement_skus_data[improvement_skus_data['SkuName'] == sample_sku].head(2).to_string(index=False))
            
            return good_skus_data, improvement_skus_data
        
        except Exception as e:
            logger.error(f"Error filtering SKUs: {str(e)}")
            raise

if __name__ == "__main__":
    api_key = os.getenv("COMET_API_KEY")
    workspace = os.getenv("COMET_WORKSPACE", "shah-noor")
    project_name = "inventory-prediction"
    experiment_id = "0fa8a4962d804de991961c0cd7c07dd6"
    features_path = "/workspaces/Machine-learning/data/transformed/monthly_features.parquet"
    
    if not api_key:
        logger.error("COMET_API_KEY environment variable not set")
        raise ValueError("COMET_API_KEY environment variable not set")
    
    filter = SKUFilter(api_key, workspace, project_name, experiment_id, threshold_perc=0.05)
    good_skus, improvement_skus = filter.filter_skus(features_path)