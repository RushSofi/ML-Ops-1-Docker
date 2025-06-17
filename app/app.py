import sys
import os
import json
import logging
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath('./src'))
from preprocessing import preprocess_data
from scorer import make_pred, get_feature_importances, plot_score_distribution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    INPUT_FILE = "/app/input/test.csv"
    OUTPUT_DIR = "/app/output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file {INPUT_FILE} not found!")
        return

    try:
        logger.info("Starting processing...")
        
        df = pd.read_csv(INPUT_FILE)
        df = preprocess_data(df)
        
        submission, scores = make_pred(df, INPUT_FILE)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        submission.to_csv(f"{OUTPUT_DIR}/sample_submission.csv", index=False)
        
        with open(f"{OUTPUT_DIR}/feature_importances.json", "w") as f:
            json.dump(get_feature_importances(), f)
        
        plot_score_distribution(scores, f"{OUTPUT_DIR}/scores_distribution.png")
        
        logger.info(f"All results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()