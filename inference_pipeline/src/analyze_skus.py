import pandas as pd
import logging
from typing import Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SKUAnalyzer:
    def __init__(self, features_path: str, good_skus_path: str, improvement_skus_path: str, priority_threshold_perc: float = 0.8, volume_threshold_perc: float = 0.8):
        """Initialize SKU analyzer with dataset paths and thresholds."""
        self.features_path = features_path
        self.good_skus_path = good_skus_path
        self.improvement_skus_path = improvement_skus_path
        self.priority_threshold_perc = priority_threshold_perc
        self.volume_threshold_perc = volume_threshold_perc
        self.df = None
        self.good_skus = None
        self.improvement_skus = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load original dataset and filtered SKUs."""
        try:
            for path in [self.features_path, self.good_skus_path, self.improvement_skus_path]:
                if not os.path.exists(path):
                    logger.error(f"File not found: {path}")
                    raise FileNotFoundError(f"File not found: {path}")
            
            self.df = pd.read_parquet(self.features_path)
            self.df['SkuName'] = self.df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            logger.debug(f"Dataset columns: {list(self.df.columns)}")
            logger.debug(f"First 5 SKUs: {self.df['SkuName'].unique()[:5].tolist()}")
            logger.info(f"Loaded original dataset with {len(self.df)} rows and {len(self.df['SkuName'].unique())} unique SKUs")
            
            self.good_skus = pd.read_csv(self.good_skus_path)
            self.good_skus['SkuName'] = self.good_skus['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            logger.debug(f"Good SKUs columns: {list(self.good_skus.columns)}")
            logger.debug(f"First 5 good SKUs: {self.good_skus['SkuName'].head().tolist()}")
            logger.info(f"Loaded good SKUs with {len(self.good_skus)} entries")
            
            self.improvement_skus = pd.read_csv(self.improvement_skus_path)
            self.improvement_skus['SkuName'] = self.improvement_skus['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            logger.debug(f"Improvement SKUs columns: {list(self.improvement_skus.columns)}")
            logger.debug(f"First 5 improvement SKUs: {self.improvement_skus['SkuName'].head().tolist()}")
            logger.info(f"Loaded improvement SKUs with {len(self.improvement_skus)} entries")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_improvement_skus(self) -> pd.DataFrame:
        """Analyze SKUs where model struggled for phase two improvements."""
        try:
            analysis = []
            # Deduplicate improvement SKUs
            unique_skus = self.improvement_skus[['SkuName', 'mae', 'performance']].drop_duplicates()
            logger.info(f"Analyzing {len(unique_skus)} unique improvement SKUs")
            
            for _, row in unique_skus.iterrows():
                sku = row['SkuName']
                sku_df = self.df[self.df['SkuName'] == sku]
                
                if sku_df.empty:
                    logger.warning(f"No data found for SKU {sku} in dataset")
                    analysis.append({
                        'SkuName': sku,
                        'data_points': 0,
                        'outliers': 0,
                        'seasonality_std': np.nan,
                        'avg_demand': np.nan,
                        'mape': np.nan,
                        'mae': row['mae'],
                        'performance': row['performance']
                    })
                    continue
                
                data_points = len(sku_df)
                Q1 = sku_df['NoOfPItems'].quantile(0.25)
                Q3 = sku_df['NoOfPItems'].quantile(0.75)
                IQR = Q3 - Q1
                outliers = len(sku_df[(sku_df['NoOfPItems'] < Q1 - 1.5 * IQR) | (sku_df['NoOfPItems'] > Q3 + 1.5 * IQR)])
                seasonality_std = sku_df['NoOfPItems'].std() if len(sku_df) > 1 else 0.0
                avg_demand = sku_df['NoOfPItems'].mean()
                
                # Handle MAPE for zero values and empty arrays
                actual = sku_df['NoOfPItems']
                predicted = np.full_like(actual, avg_demand)
                if len(actual) == 0 or (actual == 0).all():
                    mape = np.nan
                else:
                    non_zero = actual != 0
                    if non_zero.any():
                        mape = mean_absolute_percentage_error(actual[non_zero], predicted[non_zero]) * 100
                    else:
                        mape = np.nan
                
                analysis.append({
                    'SkuName': sku,
                    'data_points': data_points,
                    'outliers': outliers,
                    'seasonality_std': seasonality_std,
                    'avg_demand': avg_demand,
                    'mape': mape,
                    'mae': row['mae'],
                    'performance': row['performance']
                })
            
            analysis_df = pd.DataFrame(analysis)
            output_path = '/workspaces/Machine-learning/data/filtered/improvement_analysis.csv'
            analysis_df.to_csv(output_path, index=False)
            logger.info(f"Saved improvement SKUs analysis to {output_path}")
            
            logger.info("\nSample Improvement SKUs Analysis (first 3):")
            logger.info(analysis_df.head(3).to_string(index=False))
            
            return analysis_df
        
        except Exception as e:
            logger.error(f"Error analyzing improvement SKUs: {str(e)}")
            raise
    
    def check_volume_priority(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Check if good SKUs are high-volume/priority and vice versa."""
        try:
            sku_volume = self.df.groupby('SkuName')['NoOfPItems'].sum().reset_index()
            sku_volume.columns = ['SkuName', 'total_volume']
            sku_volume = sku_volume.sort_values('total_volume', ascending=False)
            total_volume = sku_volume['total_volume'].sum()
            sku_volume['cumulative_volume'] = sku_volume['total_volume'].cumsum() / total_volume
            high_volume_skus = sku_volume[sku_volume['cumulative_volume'] <= self.volume_threshold_perc]['SkuName'].tolist()
            logger.debug(f"High-volume SKUs: {high_volume_skus[:5]}")
            
            sku_frequency = self.df.groupby('SkuName')['OrderDate'].nunique().reset_index()
            sku_frequency.columns = ['SkuName', 'order_frequency']
            sku_frequency = sku_frequency.sort_values('order_frequency', ascending=False)
            total_frequency = sku_frequency['order_frequency'].sum()
            sku_frequency['cumulative_frequency'] = sku_frequency['order_frequency'].cumsum() / total_frequency
            high_priority_skus = sku_frequency[sku_frequency['cumulative_frequency'] <= self.priority_threshold_perc]['SkuName'].tolist()
            logger.debug(f"High-priority SKUs: {high_priority_skus[:5]}")
            
            # Deduplicate good SKUs
            good_skus_unique = self.good_skus[['SkuName', 'mae']].drop_duplicates()
            good_analysis = []
            for _, row in good_skus_unique.iterrows():
                sku = row['SkuName']
                volume = sku_volume[sku_volume['SkuName'] == sku]['total_volume'].values[0] if not sku_volume[sku_volume['SkuName'] == sku].empty else 0
                frequency = sku_frequency[sku_frequency['SkuName'] == sku]['order_frequency'].values[0] if not sku_frequency[sku_frequency['SkuName'] == sku].empty else 0
                good_analysis.append({
                    'SkuName': sku,
                    'mae': row['mae'],
                    'total_volume': volume,
                    'order_frequency': frequency,
                    'is_high_volume': sku in high_volume_skus,
                    'is_high_priority': sku in high_priority_skus
                })
            
            good_analysis_df = pd.DataFrame(good_analysis)
            good_output_path = '/workspaces/Machine-learning/data/filtered/good_skus_volume_priority.csv'
            good_analysis_df.to_csv(good_output_path, index=False)
            logger.info(f"Saved good SKUs volume/priority analysis to {good_output_path}")
            
            logger.info("\nSample Good SKUs Volume/Priority (first 3):")
            logger.info(good_analysis_df.head(3).to_string(index=False))
            
            # Deduplicate improvement SKUs
            improvement_skus_unique = self.improvement_skus[['SkuName', 'mae']].drop_duplicates()
            improvement_analysis = []
            for _, row in improvement_skus_unique.iterrows():
                sku = row['SkuName']
                volume = sku_volume[sku_volume['SkuName'] == sku]['total_volume'].values[0] if not sku_volume[sku_volume['SkuName'] == sku].empty else 0
                frequency = sku_frequency[sku_frequency['SkuName'] == sku]['order_frequency'].values[0] if not sku_frequency[sku_frequency['SkuName'] == sku].empty else 0
                improvement_analysis.append({
                    'SkuName': sku,
                    'mae': row['mae'],
                    'total_volume': volume,
                    'order_frequency': frequency,
                    'is_high_volume': sku in high_volume_skus,
                    'is_high_priority': sku in high_priority_skus
                })
            
            improvement_analysis_df = pd.DataFrame(improvement_analysis)
            improvement_output_path = '/workspaces/Machine-learning/data/filtered/improvement_skus_volume_priority.csv'
            improvement_analysis_df.to_csv(improvement_output_path, index=False)
            logger.info(f"Saved improvement SKUs volume/priority analysis to {improvement_output_path}")
            
            logger.info("\nSample Improvement SKUs Volume/Priority (first 3):")
            logger.info(improvement_analysis_df.head(3).to_string(index=False))
            
            return good_analysis_df, improvement_analysis_df
        
        except Exception as e:
            logger.error(f"Error checking volume and priority: {str(e)}")
            raise

def main():
    """Run SKU analysis."""
    try:
        features_path = '/workspaces/Machine-learning/data/transformed/monthly_features.parquet'
        good_skus_path = '/workspaces/Machine-learning/data/filtered/good_skus.csv'
        improvement_skus_path = '/workspaces/Machine-learning/data/filtered/improvement_skus.csv'
        
        analyzer = SKUAnalyzer(features_path, good_skus_path, improvement_skus_path)
        analyzer.analyze_improvement_skus()
        analyzer.check_volume_priority()
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()