# User Manual

## 1. Setup
Ensure you have installed the requirements:
```bash
pip install -r requirements.txt
```

## 2. Generating Data
If starting fresh, generate synthetic data:
```bash
python src/generate_synthetic_data.py
```

## 3. Training the Model
Train the machine learning model:
```bash
python src/train.py
```
This will save the best model to `models/best_model.pkl`.

## 4. Using the Forecast App
Launch the application:
```bash
streamlit run src/forecast_app.py
```

### App Features:
- **Select Store**: Choose the store ID from the sidebar.
- **Select Product**: Choose the product ID from the sidebar.
- **Forecast Horizon**: Adjust the slider to forecast 1 to 30 days ahead.
- **Visualizations**: View historical sales and future demand plots.
- **Recommendations**: See suggested reorder quantities based on safety stock calculations.

## 5. Example Scenario
**Goal**: Forecast demand for Product 101 at Store 1 for next 7 days.
1. Open the app.
2. Select "1" in "Select Store".
3. Select "101" in "Select Product".
4. Set "Forecast Horizon" to 7.
5. View the "Forecast for next 7 days" table and the "Inventory Recommendations".
