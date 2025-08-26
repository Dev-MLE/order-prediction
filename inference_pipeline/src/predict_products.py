import requests
import pandas as pd
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_products(products: List[Dict], endpoint: str = "https://ominous-carnival-8000.app.github.dev/predict") -> pd.DataFrame:
    """Make predictions for a list of products."""
    results = []
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    for product in products:
        try:
            response = requests.post(endpoint, json=product, headers=headers)
            response.raise_for_status()
            result = response.json()
            results.append({
                "sku_name": result["sku_name"],
                "predicted_demand": result["predicted_demand"],
                "model_type": result["model_type"],
                "status": "success"
            })
            logger.info(f"Prediction for {product['sku_name']}: {result['predicted_demand']:.4f} ({result['model_type']})")
        except Exception as e:
            logger.error(f"Prediction failed for {product['sku_name']}: {str(e)}")
            results.append({
                "sku_name": product["sku_name"],
                "predicted_demand": None,
                "model_type": None,
                "status": f"error: {str(e)}"
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Customize with your products
    products = [
        {
            "sku_name": "ALB2-T 400T cobas c111",
            "supplier_name": "Supplier A",
            "product_category": "Medical Equipment",
            "prediction_horizon": 1
        },
        {
            "sku_name": "I.V Cannula 22#",
            "supplier_name": "Supplier A",
            "product_category": "Medical Equipment",
            "prediction_horizon": 1,
            "estimated_demand": 500.0
        },
        {
            "sku_name": "New Product XYZ",
            "supplier_name": "Supplier B",
            "product_category": "Medical Supplies",
            "prediction_horizon": 1,
            "estimated_demand": 1000.0
        },
        # Add your SKUs here, e.g.:
        {
            "sku_name": "Syringe 5ml",
            "supplier_name": "Supplier C",
            "product_category": "Medical Supplies",
            "prediction_horizon": 1,
            "estimated_demand": 200.0
        },
        {
            "sku_name": "Gloves Nitrile",
            "supplier_name": "Supplier D",
            "product_category": "Medical Supplies",
            "prediction_horizon": 1
        }
    ]
    
    # Load improvement SKUs (optional)
    improvement_skus_path = "/workspaces/Machine-learning/inference_pipeline/data/filtered/improvement_skus.csv"
    if Path(improvement_skus_path).exists():
        improvement_skus = pd.read_csv(improvement_skus_path)['SkuName'].tolist()
        products.extend([
            {
                "sku_name": sku,
                "supplier_name": "Unknown",
                "product_category": "Unknown",
                "prediction_horizon": 1
            } for sku in improvement_skus[:5]  # Limit to 5 for testing
        ])
    
    results_df = predict_products(products)
    print(results_df)
    results_df.to_csv("/workspaces/Machine-learning/inference_pipeline/results/predictions.csv", index=False)