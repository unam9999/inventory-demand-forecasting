import pandas as pd
import numpy as np

def create_features(df):
    """
    Create features for time series forecasting.
    
    Args:
        df (pd.DataFrame): Dataframe with basic columns.
        
    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    df = df.copy()
    
    # Sort to ensure rolling calculations are correct
    df = df.sort_values(['store_id', 'product_id', 'date'])
    
    # 1. Date Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 2. Lag Features (Past demand)
    # We need to group by store and product to shift correctly
    # Lags: 1 day, 7 days (same day last week), 14 days
    # Note: For real forecasting, we must ensure we don't leak future info.
    # If we forecast 7 days ahead, we can't use lag_1 for the 7th day unless we use recursive forecasting.
    # For this project, we'll assume we are forecasting 1 day ahead or using a direct strategy where we have these available,
    # OR we are training for 1-step ahead.
    
    # To be safe for a general model, let's use lags that would be available if we run daily.
    
    grouped = df.groupby(['store_id', 'product_id'])
    
    df['lag_1'] = grouped['units_sold'].shift(1)
    df['lag_7'] = grouped['units_sold'].shift(7)
    df['lag_14'] = grouped['units_sold'].shift(14)
    
    # 3. Rolling Window Features
    # Mean and Std of last 7 and 30 days
    # We use shift(1) before rolling to avoid data leakage (include yesterday, not today)
    
    df['rolling_mean_7'] = grouped['units_sold'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = grouped['units_sold'].shift(1).rolling(window=7).std()
    
    df['rolling_mean_30'] = grouped['units_sold'].shift(1).rolling(window=30).mean()
    df['rolling_std_30'] = grouped['units_sold'].shift(1).rolling(window=30).std()
    
    # 4. Interaction Features (optional, simple ones)
    df['price_log'] = np.log1p(df['price'])
    
    # Drop rows with NaNs created by lags/rolling (first few days)
    df = df.dropna()
    
    return df

def encode_categorical(df, categorical_cols=None):
    """
    One-hot encode categorical features.
    
    Args:
        df (pd.DataFrame): Dataframe.
        categorical_cols (list): List of columns to encode.
        
    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    if categorical_cols is None:
        categorical_cols = ['store_id', 'product_id', 'day_of_week', 'month']
        
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df
