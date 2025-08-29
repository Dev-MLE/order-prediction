from dataclasses import dataclass
from dotenv import load_dotenv
import logging
import os

# Load .env
load_dotenv()

@dataclass
class FeaturePipelineConfig:
    log_level: str = "INFO"

    # Data loading
    use_db: bool = False
    # for running locally use full path, this is the path for github actions.
    # You can add this path directly in workflow file for better consistency.
    data_path: str = "data/raw/SundasPoDetail.xlsx"

    # Build DB connection string from env vars
    db_user: str = os.getenv("DB_USER", "user")
    db_pass: str = os.getenv("DB_PASS", "pass")
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str = os.getenv("DB_NAME", "sales_db")

    db_connection: str = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    db_query: str = "SELECT * FROM sales_data"

    # Preprocessing
    aggregation_freq: str = "M"

    # Feature Engineering
    lag_periods: int = 3
    rolling_window: int = 3

    # Output
    use_hopsworks: bool = False # Toggle to use/avoid hopsworks storage.
    output_path: str = "data/transformed/monthly_features.parquet"
   # output_filename: str = "monthly_features.parquet"

    # Hopsworks
    hopsworks_api_key: str = os.getenv("HOPSWORKS_API_KEY")
    feature_group_name: str = "demand_records"
    feature_group_version: int = 1

def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("feature_pipeline.log")
        ]
    )

