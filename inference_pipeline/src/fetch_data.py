
import pandas as pd
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PredictionDataFetcher:
    """Fetch and prepare data for predictions, including for unknown SKUs."""
    
    def __init__(self, features_path: str, good_skus_path: str):
        self.features_path = features_path
        self.good_skus_path = good_skus_path
        self.df = None
        self.good_skus = None
        self.has_supplier = False
        self.load_data()
    
    def load_data(self) -> None:
        """Load features and good SKUs."""
        try:
            self.df = pd.read_parquet(self.features_path)
            self.df['SkuName'] = self.df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
            self.has_supplier = 'SupplierName' in self.df.columns
            logger.info(f"Loaded features dataset with {len(self.df)} rows and {len(self.df['SkuName'].unique())} unique SKUs. SupplierName column: {self.has_supplier}")
            
            self.good_skus = pd.read_csv(self.good_skus_path)['SkuName'].unique().tolist()
            logger.info(f"Loaded {len(self.good_skus)} good SKUs for prediction")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_prediction_input(self, sku_name: str, supplier_name: Optional[str] = None, 
                               product_category: Optional[str] = None, estimated_demand: Optional[float] = None,
                               prediction_horizon: int = 1) -> Dict:
        """Prepare input features for prediction, handling unknown SKUs."""
        sku_name_clean = sku_name.strip().replace(r'\s+', ' ')
        is_known_sku = sku_name_clean in self.good_skus
        
        if is_known_sku:
            sku_df = self.df[self.df['SkuName'] == sku_name_clean]
            if sku_df.empty:
                logger.error(f"No data found for known SKU {sku_name_clean}")
                raise ValueError(f"No data found for SKU {sku_name_clean}")
            
            last_date = pd.to_datetime(sku_df['OrderDate'].max())
            lag1 = sku_df['NoOfPItems'].iloc[-1]
            lag2 = sku_df['NoOfPItems'].iloc[-2] if len(sku_df) > 1 else lag1
            rolling_mean = sku_df['NoOfPItems'].mean()
            rolling_std = sku_df['NoOfPItems'].std() if len(sku_df) > 1 else 0.0
        else:
            logger.warning(f"SKU {sku_name_clean} not in good SKUs, estimating features")
            if supplier_name and self.has_supplier:
                supplier_df = self.df[self.df['SupplierName'] == supplier_name]
                if not supplier_df.empty:
                    lag1 = supplier_df['NoOfPItems'].mean()
                    lag2 = lag1
                    rolling_mean = supplier_df['NoOfPItems'].mean()
                    rolling_std = supplier_df['NoOfPItems'].std() if len(supplier_df) > 1 else 0.0
                else:
                    logger.warning(f"No supplier data for {supplier_name}, using defaults")
                    lag1 = estimated_demand or 100.0
                    lag2 = lag1
                    rolling_mean = lag1
                    rolling_std = 0.0
            else:
                logger.warning(f"No SupplierName column or supplier_name provided, using defaults")
                lag1 = estimated_demand or 100.0
                lag2 = lag1
                rolling_mean = lag1
                rolling_std = 0.0
            last_date = datetime.now()
        
        # Generate future dates for monthly predictions
        future_dates = [last_date + relativedelta(months=i+1) for i in range(prediction_horizon)]
        
        # Prepare features for future dates
        future_features = []
        for date in future_dates:
            month = date.month
            year = date.year
            quarter = (month - 1) // 3 + 1
            is_holiday = 0  # Placeholder, replace with holiday logic if available
            future_features.append({
                'OrderDate': date,
                'Month': month,
                'Year': year,
                'Quarter': quarter,
                'IsHoliday': is_holiday,
                'Lag1': lag1,
                'Lag2': lag2,
                'RollingMean': rolling_mean,
                'RollingStd': rolling_std,
                'ProductCategory': product_category or 'Unknown'
            })
        
        future_df = pd.DataFrame(future_features)
        logger.info(f"Prepared input features for SKU {sku_name_clean} (known: {is_known_sku}) for {prediction_horizon} months")
        return future_df.to_dict(orient='records')

if __name__ == "__main__":
    features_path = '/workspaces/Machine-learning/data/transformed/monthly_features.parquet'
    good_skus_path = '/workspaces/Machine-learning/data/filtered/good_skus.csv'
    
    fetcher = PredictionDataFetcher(features_path, good_skus_path)
    # Example: Known SKU
    prediction_input = fetcher.prepare_prediction_input(
        sku_name="ALT - 400T Cobas c111",
        prediction_horizon=1
    )
    print("Known SKU:", prediction_input)
    
    # Example: Unknown SKU
    prediction_input = fetcher.prepare_prediction_input(
        sku_name="New Product XYZ",
        supplier_name="Supplier A",
        product_category="Medical Equipment",
        estimated_demand=500.0,
        prediction_horizon=1
    )
    print("Unknown SKU:", prediction_input)