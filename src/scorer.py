import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD =  0.97781  

logger.info('Importing pretrained LightGBM model...')
model = lgb.Booster(model_file='./models/model.txt')
model_th = DEFAULT_THRESHOLD

def make_pred(dt, path_to_file):
    required_features = model.feature_name()
    missing = set(required_features) - set(dt.columns)
    extra = set(dt.columns) - set(required_features)
    
    if missing:
        logger.warning(f"Missing features: {missing}")
        for feat in missing:
            dt[feat] = 0  
    
    if extra:
        logger.warning(f"Extra features: {extra}")
        dt = dt[required_features] 
    
    scores = model.predict(dt)
    predictions = (scores >= model_th).astype(int)
    
    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': predictions
    })
    
    logger.info('Prediction complete for file: %s', path_to_file)
    return submission, scores

def get_feature_importances(top_n=5):
    importance = model.feature_importance(importance_type='gain')
    features = model.feature_name()
    return dict(sorted(zip(features, importance), 
               key=lambda x: x[1], reverse=True)[:top_n])

def plot_score_distribution(scores, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title('Distribution of Prediction Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()