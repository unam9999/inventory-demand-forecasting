import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    RAW_DATA_FILE, MODEL_FILE, SCALER_FILE, 
    TRAIN_DATA_FILE, TEST_DATA_FILE, MODELS_DIR
)
from src.data_loader import load_data
from src.preprocessing import preprocess_data, split_data
from src.feature_engineering import create_features, encode_categorical
from src.models import get_random_forest_model

def train_model():
    print("Starting training pipeline...")
    
    # 1. Load Data
    df = load_data(RAW_DATA_FILE)
    
    # 2. Preprocess
    df = preprocess_data(df)
    
    # 3. Feature Engineering
    df = create_features(df)
    
    # Keep original IDs for reference/evaluation
    df['store_id_orig'] = df['store_id']
    df['product_id_orig'] = df['product_id']
    
    df = encode_categorical(df)
    
    # 4. Split Data
    train_df, test_df = split_data(df)
    
    # Save processed data for evaluation/app
    os.makedirs(os.path.dirname(TRAIN_DATA_FILE), exist_ok=True)
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    print(f"Saved processed data to {TRAIN_DATA_FILE} and {TEST_DATA_FILE}")
    
    # Prepare X and y
    target = 'units_sold'
    drop_cols = ['date', 'units_sold', 'on_hand_inventory', 'stockout_flag', 'store_id_orig', 'product_id_orig'] 
    # Note: We drop 'date' as models don't handle datetime objects directly usually
    # We drop 'on_hand_inventory' and 'stockout_flag' as they are not known in future (or are state variables)
    # Actually, on_hand_inventory is known for today, but for forecasting tomorrow we might know it.
    # However, let's assume we rely on demand features.
    
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]
    
    print(f"Training features: {X_train.columns.tolist()}")
    
    # 5. Model Training (Random Forest)
    print("Training Random Forest model...")
    rf = get_random_forest_model()
    
    # Simple Grid Search (Optional, kept small for speed)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    # Use TimeSeriesSplit for CV
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # 6. Evaluation on Test Set (Preliminary)
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # 7. Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    # We don't strictly need a scaler for RF, but if we used linear models we would.
    # Saving a dummy scaler or just nothing if not needed.
    # Let's save the column names to ensure consistency during inference
    joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_DIR, 'model_columns.pkl'))

if __name__ == "__main__":
    train_model()
