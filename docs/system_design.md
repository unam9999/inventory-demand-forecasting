# System Design Document

## 1. Introduction
The Inventory Demand Forecasting System is designed to predict future sales for retail products to optimize inventory levels. It uses machine learning to analyze historical sales data and generate daily demand forecasts.

## 2. Architecture Overview
The system follows a modular architecture with the following layers:

1.  **Data Layer**: Handles data ingestion and storage.
    - Raw CSV data storage.
    - Data loading modules (`data_loader.py`).
2.  **Processing Layer**: Cleans and transforms data.
    - Preprocessing (`preprocessing.py`).
    - Feature Engineering (`feature_engineering.py`).
3.  **Modeling Layer**: Trains and manages ML models.
    - Model definitions (`models.py`).
    - Training pipeline (`train.py`).
4.  **Application Layer**: User interface for forecasting.
    - Streamlit App (`forecast_app.py`).

## 3. Data Flow
1.  **Ingestion**: Raw sales data is read from `data/raw/historical_sales_sample.csv`.
2.  **Preprocessing**: Missing values are handled, and data is split into training/testing sets.
3.  **Feature Engineering**: Time-series features (lags, rolling means) and calendar features are generated.
4.  **Training**: A Random Forest Regressor is trained on the processed data.
5.  **Inference**: The model predicts future demand based on recent history.
6.  **Visualization**: Results are displayed in the Streamlit app.

## 4. Components

### 4.1 Data Loader
Responsible for reading CSV files and parsing dates.

### 4.2 Preprocessor
Handles data cleaning and train/test splitting.

### 4.3 Feature Engineer
Creates lag features (1, 7, 14 days), rolling statistics (7, 30 days), and encodes categorical variables.

### 4.4 Model
Uses `RandomForestRegressor` from scikit-learn. Hyperparameters are tuned using `GridSearchCV`.

### 4.5 Forecast App
Interactive dashboard built with Streamlit allowing users to select store/product and view forecasts.
