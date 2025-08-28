from settings import FeaturePipelineConfig
from pipeline import FeaturePipeline
import logging

def main():
    """Run the feature pipeline."""
    try:
        config = FeaturePipelineConfig()
        #    use_db=False,       # Toggle True/False
        #    use_hopsworks=True,  # Toggle True/False
        #)
        
        pipeline = FeaturePipeline(config)
        result = pipeline.run()
        if result is not None:
            logging.info("Feature pipeline completed successfully")
        else:
            logging.error("Feature pipeline failed")
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
