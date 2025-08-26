import pandas as pd
import numpy as np
from comet_ml import API, Experiment
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model():
    """Evaluate the performance of the final_general_model on the test set."""
    try:
        # Initialize Comet.ml experiment
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="inventory-prediction",
            workspace=os.getenv("COMET_WORKSPACE", "shah-noor")
        )
        experiment.set_name("evaluate_final_general_model")

        # Load model
        model_path = Path('/workspaces/Machine-learning/models/final_general_model.pkl')
        if not model_path.exists():
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load features and good SKUs
        features_path = "/workspaces/Machine-learning/inference_pipeline/data/transformed/monthly_features.parquet"
        good_skus_path = "/workspaces/Machine-learning/inference_pipeline/data/filtered/good_skus.csv"
        
        df = pd.read_parquet(features_path)
        df['SkuName'] = df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
        logger.info(f"Loaded features dataset with {len(df)} rows and {len(df['SkuName'].unique())} unique SKUs")
        
        good_skus = pd.read_csv(good_skus_path)['SkuName'].unique().tolist()
        logger.info(f"Loaded {len(good_skus)} good SKUs")

        # Filter for good SKUs
        good_df = df[df['SkuName'].isin(good_skus)]
        if len(good_df) < 1:
            logger.error("No data for good SKUs")
            raise ValueError("No data for good SKUs")

        # Prepare features and target
        X = good_df[['Month', 'Year', 'Quarter', 'IsHoliday', 'Lag1', 'Lag2', 'RollingMean', 'RollingStd']]
        y = good_df['NoOfPItems']

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if len(X_test) < 1:
            logger.error("No test data available")
            raise ValueError("No test data available")

        # Make predictions
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Test MAE: {mae:.4f}")

        # Log metrics to Comet.ml
        experiment.log_metric("test_mae", mae)
        experiment.log_other("num_test_samples", len(X_test))

        # Log sample predictions
        sample_predictions = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred
        }).head(10)
        experiment.log_table("sample_predictions.csv", sample_predictions)

        logger.info("Evaluation completed and logged to Comet.ml")
        experiment.end()

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        experiment.log_other("evaluation_error", str(e))
        experiment.end()
        raise

if __name__ == "__main__":
    evaluate_model()