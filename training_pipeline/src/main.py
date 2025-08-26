import logging
import os
from dotenv import load_dotenv
from config import TrainingPipelineConfig
from pipeline import TrainingPipeline
from combined_model import ModelCombiner

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the training pipeline (if needed) and combine models."""
    try:
        # Check if models exist in the specified experiment
        api_key = os.getenv("COMET_API_KEY")
        workspace = os.getenv("COMET_WORKSPACE", "shah-noor")
        project_name = "inventory-prediction"
        features_path = "/workspaces/Machine-learning/inference_pipeline/data/transformed/monthly_features.parquet"
        good_skus_path = "/workspaces/Machine-learning/inference_pipeline/data/filtered/good_skus.csv"
        experiment_id = ""  # Updated to match logs
        
        logger.info(f"Attempting to use experiment ID: {experiment_id}")
        combiner = ModelCombiner(api_key, workspace, project_name, features_path, good_skus_path, experiment_id)
        combiner.fetch_models_and_performance()
        
        if not combiner.models:
            logger.info("No models found in specified experiment, running training pipeline")
            config = TrainingPipelineConfig()
            pipeline = TrainingPipeline(config)
            training_results = pipeline.run()
            if not training_results:
                logger.error("Training pipeline failed: No results returned")
                raise ValueError("Training pipeline failed")
            logger.info(f"Training pipeline completed successfully with results for {len(training_results)} SKUs")
            # Fetch models from the new experiment
            combiner = ModelCombiner(api_key, workspace, project_name, features_path, good_skus_path)
            combiner.fetch_models_and_performance()
        
        final_model = combiner.combine_models()
        combiner.upload_final_model(final_model)
    
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()