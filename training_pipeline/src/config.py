from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TrainingPipelineConfig:
    comet_api_key: str = os.getenv("COMET_API_KEY")
    workspace: str = os.getenv("COMET_WORKSPACE", "shah-noor")
    project_name: str = "inventory-prediction"
    features_path: str = "/workspaces/Machine-learning/inference_pipeline/data/transformed/monthly_features.parquet"
    good_skus_path: str = "/workspaces/Machine-learning/inference_pipeline/data/filtered/good_skus.csv"
    model_dir: str = "/workspaces/Machine-learning/models"