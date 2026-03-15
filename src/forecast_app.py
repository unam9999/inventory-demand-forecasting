import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    MODEL_FILE, MODELS_DIR, RAW_DATA_FILE
)
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features, encode_categorical

# Page config
st.set_page_config(page_title="Inventory Demand Forecasting", layout="wide")

@st.cache_resource
def load_model_and_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None, None
    model = joblib.load(MODEL_FILE)
    model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.pkl'))
    return model, model_columns

@st.cache_data
def load_and_prep_data():
    df = load_data(RAW_DATA_FILE)
    df = preprocess_data(df)
    # We need features for the whole dataset to show history
    df_features = create_features(df)
    df_encoded = encode_categorical(df_features)
    return df_features, df_encoded

def main():
    st.title("Inventory Demand Forecasting System")
    
    model, model_columns = load_model_and_artifacts()
    if model is None:
        st.error("Model not found. Please run training first.")
        return
        
    df, df_encoded = load_and_prep_data()
    
    # Sidebar controls
    st.sidebar.header("Forecast Settings")
    
    # Select Store and Product
    store_ids = sorted(df['store_id'].unique())
    product_ids = sorted(df['product_id'].unique())
    
    selected_store = st.sidebar.selectbox("Select Store", store_ids)
    selected_product = st.sidebar.selectbox("Select Product", product_ids)
    
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 7)
    
    # Filter data for selected item
    mask = (df['store_id'] == selected_store) & (df['product_id'] == selected_product)
    item_history = df[mask].copy()
    
    if item_history.empty:
        st.warning("No data found for this combination.")
        return
        
    # Display historical data
    st.subheader(f"Historical Demand: Store {selected_store}, Product {selected_product}")
    
    # Plot history
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(item_history['date'], item_history['units_sold'], label='Historical Sales')
    ax.set_title("Historical Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.grid(True)
    st.pyplot(fig)
    
    # Forecasting Logic
    # For a real forecast, we need future features.
    # We will generate future dates and features.
    # Note: Lags are tricky for multi-step forecasting without recursion.
    # Here we will implement a simple recursive strategy or just use the last known values 
    # (which is a simplification, but acceptable for this demo).
    
    last_date = item_history['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    
    st.subheader(f"Forecast for next {forecast_horizon} days")
    
    future_preds = []
    
    # We need the last part of data to generate features for the first future day
    # Then we append the prediction and re-generate features for the next day (recursive)
    
    # Get the full encoded dataframe for this item to pick up rolling windows correctly
    # Use the mask from df to filter df_encoded (indices should align)
    item_encoded = df_encoded.loc[mask].copy()
    
    # We need to append new rows one by one
    current_df = item_encoded.copy()
    
    # Columns that we need to fill manually for future
    # date, store_id, product_id, price, promotion, holiday, etc.
    # Lags and rolling will be calculated by create_features logic if we re-run it, 
    # but create_features expects raw-ish data.
    # To simplify, let's just use the model on the features we can construct.
    # Constructing features recursively is complex in a short script.
    # ALTERNATIVE: Just use the test set logic (if we have future test data).
    # BUT the user wants a forecast for the future.
    
    # SIMPLIFICATION for the App:
    # We will just take the last available row of features and repeat it/shift it slightly 
    # or just use the 'day_of_week' etc which we know.
    # For lags, we will use the last known values.
    
    # Let's try to do it somewhat correctly:
    # We'll take the last 30 days of history, append a new row, re-calculate features, predict, fill value, repeat.
    
    # We need a mini-pipeline for a single row
    
    # 1. Get recent history (raw)
    recent_history = item_history.tail(60).copy() # Enough for 30-day rolling
    
    for date in future_dates:
        # Create a new row
        new_row = {
            'date': date,
            'store_id': selected_store,
            'product_id': selected_product,
            'on_hand_inventory': 0, # Irrelevant for demand prediction usually
            'units_sold': np.nan, # To be predicted
            'price': recent_history.iloc[-1]['price'], # Assume price stays same
            'promotion': 0, # Assume no promo
            'holiday': 0, # Assume no holiday (or calculate)
            'lead_time_days': 0,
            'stockout_flag': 0
        }
        
        # Append to history
        recent_history = pd.concat([recent_history, pd.DataFrame([new_row])], ignore_index=True)
        
        # Re-calculate features
        # We need to call create_features on this small df
        # Note: create_features expects 'units_sold' to be present for lags.
        # It handles NaNs by shifting, but if the target is NaN, the lag will be NaN next time.
        # So we must fill 'units_sold' with prediction after predicting.
        
        features_df = create_features(recent_history)
        features_encoded = encode_categorical(features_df)
        
        # Get the last row (the one we want to predict)
        # Ensure all model columns are present
        last_row = features_encoded.iloc[[-1]].copy()
        
        # Add missing columns (from one-hot encoding that might be missing in this small slice)
        for col in model_columns:
            if col not in last_row.columns:
                last_row[col] = 0
                
        # Select model columns
        X_future = last_row[model_columns]
        
        # Predict
        pred = model.predict(X_future)[0]
        pred = max(0, pred) # No negative demand
        
        future_preds.append(pred)
        
        # Fill the prediction back into recent_history for the next step's lag
        recent_history.iloc[-1, recent_history.columns.get_loc('units_sold')] = pred
        
    # Display Forecast
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Demand': future_preds
    })
    
    st.dataframe(forecast_df)
    
    # Plot Forecast
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    # Show last 30 days of history + forecast
    plot_history = item_history.tail(30)
    ax2.plot(plot_history['date'], plot_history['units_sold'], label='History')
    ax2.plot(forecast_df['Date'], forecast_df['Predicted Demand'], label='Forecast', linestyle='--', color='red')
    ax2.set_title("Demand Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    
    # Inventory Advice
    st.subheader("Inventory Recommendations")
    total_demand = sum(future_preds)
    safety_stock = np.std(item_history['units_sold']) * 1.65 # 95% service level approx
    reorder_point = total_demand + safety_stock
    
    st.write(f"**Total Predicted Demand (next {forecast_horizon} days):** {total_demand:.2f}")
    st.write(f"**Recommended Safety Stock:** {safety_stock:.2f}")
    st.write(f"**Suggested Reorder Quantity (to cover period + safety):** {reorder_point:.2f}")

if __name__ == "__main__":
    main()
