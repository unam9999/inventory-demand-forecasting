# Inventory Demand Forecasting System

## Project Overview
This project implements an end-to-end machine learning system for forecasting inventory demand. It predicts future sales for multiple products across different stores to optimize stock levels.

## Folder Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `src/`: Source code for data loading, processing, modeling, and forecasting.
- `models/`: Saved trained models and scalers.
- `reports/`: Generated reports and figures.
- `tests/`: Unit tests.
- `docs/`: Documentation.

## Setup Instructions

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate Data**:
    Run the data generation script (to be implemented) or ensure `data/raw/historical_sales_sample.csv` exists.

4.  **Run Training**:
    ```bash
    python src/train.py
    ```

5.  **Run Evaluation**:
    ```bash
    python src/evaluate.py
    ```

6.  **Run Forecast App**:
    ```bash
    streamlit run src/forecast_app.py
    ```

## Testing
Run unit tests using pytest:
```bash
pytest tests/
```
