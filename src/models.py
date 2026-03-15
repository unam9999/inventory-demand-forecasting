import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class NaiveBaseline(BaseEstimator, RegressorMixin):
    """
    Naive baseline: Predicts tomorrow's demand to be the same as today's (lag_1).
    """
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        # Assumes 'lag_1' is present in X
        if 'lag_1' not in X.columns:
            raise ValueError("Feature 'lag_1' required for NaiveBaseline")
        return X['lag_1'].values

class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Moving Average baseline: Predicts demand as the average of last 7 days.
    """
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        # Assumes 'rolling_mean_7' is present in X
        if 'rolling_mean_7' not in X.columns:
            raise ValueError("Feature 'rolling_mean_7' required for MovingAverageBaseline")
        return X['rolling_mean_7'].values

def get_random_forest_model():
    """
    Returns a Random Forest Regressor model.
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
