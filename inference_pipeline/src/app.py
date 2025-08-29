from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from comet_ml import API

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
model_path = Path('/workspaces/Machine-learning/models/final_general_model.pkl')
features_path = "/workspaces/Machine-learning/inference_pipeline/data/transformed/monthly_features.parquet"

# Load model: Local first, Comet fallback
general_model = None
if model_path.exists():
    general_model = joblib.load(model_path)
    logger.info(f"Loaded general model from {model_path}")
else:
    logger.warning(f"Local model not found at {model_path}, attempting to fetch from Comet registry")
    try:
        comet_api_key = os.getenv("COMET_API_KEY")
        workspace = os.getenv("COMET_WORKSPACE_NAME")
        model_name = os.getenv("COMET_MODEL_NAME", "general-inventory-model")
        model_version = os.getenv("COMET_MODEL_VERSION", "1.0.0")

        if not comet_api_key or not workspace:
            raise ValueError("Missing COMET_API_KEY or COMET_WORKSPACE_NAME in environment variables")

        api = API(api_key=comet_api_key)

        # Download model version from registry into local path
        downloaded_path = api.download_registry_model(
            workspace=workspace,
            registry_name=model_name,
            version=model_version,
            output_path=str(model_path.parent),
            expand=True  # handles zip/tar archives
        )

        # Ensure consistent filename
        os.replace(downloaded_path, model_path)

        logger.info(f"Downloaded model {model_name}:{model_version} from Comet and saved as {model_path}")

        # Load into memory
        general_model = joblib.load(model_path)
        logger.info("Loaded general model from Comet registry")

    except Exception as e:
        logger.error(f"Failed to fetch model from Comet: {str(e)}")
        raise FileNotFoundError("General model not available locally or in Comet registry.")

# Load features
features_df = pd.read_parquet(features_path)
features_df['SkuName'] = features_df['SkuName'].str.strip().str.replace(r'\s+', ' ', regex=True)
logger.info(f"Loaded features dataset with {len(features_df)} rows and {len(features_df['SkuName'].unique())} unique SKUs")

# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application")
    yield
    logger.info("Shutting down FastAPI application")

app = FastAPI(title="Inventory Prediction API", version="1.0.0", lifespan=lifespan)

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    sku_name: str
    date_to_predict: str  # Format: YYYY-MM-DD
    supplier_id: str | None = None
    estimated_demand: float | None = None

def get_features_for_sku(sku: str, predict_date: datetime, estimated_demand: float | None) -> dict:
    """Generate features for prediction."""
    sku_data = features_df[features_df['SkuName'] == sku]
    if len(sku_data) > 0:
        lag1 = estimated_demand if estimated_demand is not None else sku_data['NoOfPItems'].iloc[-1] if len(sku_data) > 0 else 0.0
        lag2 = sku_data['NoOfPItems'].iloc[-2] if len(sku_data) > 1 else 0.0
        rolling_mean = sku_data['NoOfPItems'].tail(3).mean() if len(sku_data) >= 3 else 0.0
        rolling_std = sku_data['NoOfPItems'].tail(3).std() if len(sku_data) >= 3 else 0.0
    else:
        lag1 = estimated_demand if estimated_demand is not None else 0.0
        lag2 = 0.0
        rolling_mean = 0.0
        rolling_std = 0.0
    
    return {
        'Month': predict_date.month,
        'Year': predict_date.year,
        'Quarter': (predict_date.month - 1) // 3 + 1,
        'IsHoliday': 0,
        'Lag1': lag1,
        'Lag2': lag2,
        'RollingMean': rolling_mean,
        'RollingStd': rolling_std
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        sku_name_clean = request.sku_name.strip().replace(r'\s+', ' ')
        predict_date = datetime.strptime(request.date_to_predict, '%Y-%m-%d')
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Using general model for SKU: {sku_name_clean}")
        features = get_features_for_sku(sku_name_clean, predict_date, request.estimated_demand)
        input_df = pd.DataFrame([features])
        prediction = general_model.predict(input_df)[0]
        prediction = max(0.0, float(prediction))
        
        logger.info(f"Prediction for SKU {sku_name_clean} on {request.date_to_predict}: {prediction:.4f}")
        
        return {
            "sku/product_id": sku_name_clean,
            "current_date": current_date,
            "predicted_date": request.date_to_predict,
            "no_of_orders_predicted": prediction
        }
    
    except ValueError as e:
        logger.error(f"Invalid date format for SKU {request.sku_name}: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    except Exception as e:
        logger.error(f"Prediction failed for SKU {request.sku_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
