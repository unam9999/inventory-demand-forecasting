import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    MODEL_FILE, TEST_DATA_FILE, FIGURES_DIR, MODELS_DIR
)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model():
    print("Starting evaluation...")
    
    # Load data and model
    if not os.path.exists(TEST_DATA_FILE):
        raise FileNotFoundError("Test data not found. Run training first.")
        
    test_df = pd.read_csv(TEST_DATA_FILE)
    model = joblib.load(MODEL_FILE)
    model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.pkl'))
    
    # Prepare X and y
    target = 'units_sold'
    X_test = test_df[model_columns]
    y_test = test_df[target]
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Save metrics to a text file for the report
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(os.path.join(FIGURES_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        
    # Visualizations
    
    # 1. Actual vs Predicted (Sample)
    # Let's pick one store-product combination
    sample_store = test_df['store_id_orig'].iloc[0]
    sample_product = test_df['product_id_orig'].iloc[0]
    
    mask = (test_df['store_id_orig'] == sample_store) & (test_df['product_id_orig'] == sample_product)
    sample_df = test_df[mask].copy()
    sample_pred = y_pred[mask]
    
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(sample_df['date']), sample_df['units_sold'], label='Actual', marker='o')
    plt.plot(pd.to_datetime(sample_df['date']), sample_pred, label='Predicted', linestyle='--', marker='x')
    plt.title(f'Actual vs Predicted Demand (Store {sample_store}, Product {sample_product})')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'actual_vs_predicted.png'))
    plt.close()
    
    # 2. Error Distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.savefig(os.path.join(FIGURES_DIR, 'error_distribution.png'))
    plt.close()
    
    # 3. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = 10
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(top_n), importances[indices[:top_n]], align='center')
        plt.xticks(range(top_n), [X_test.columns[i] for i in indices[:top_n]], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'))
        plt.close()
        
    print(f"Plots saved to {FIGURES_DIR}")

if __name__ == "__main__":
    evaluate_model()
