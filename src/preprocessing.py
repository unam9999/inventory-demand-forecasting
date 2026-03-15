import pandas as pd
import numpy as np
from src.config import TEST_SIZE_DAYS

def preprocess_data(df):
    """
    Clean and preprocess the raw data.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df = df.copy()
    
    # Handle missing values
    # In our synthetic data, we shouldn't have many, but let's be safe
    # Forward fill for price and inventory, 0 for others if any
    df['price'] = df['price'].ffill()
    df['on_hand_inventory'] = df['on_hand_inventory'].ffill()
    df = df.fillna(0)
    
    # Filter invalid rows
    df = df[df['units_sold'] >= 0]
    df = df[df['price'] > 0]
    
    return df

def split_data(df, test_days=TEST_SIZE_DAYS):
    """
    Split data into training and testing sets based on time.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        test_days (int): Number of days to include in test set.
        
    Returns:
        tuple: (train_df, test_df)
    """
    max_date = df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)
    
    train_df = df[df['date'] <= cutoff_date].copy()
    test_df = df[df['date'] > cutoff_date].copy()
    
    print(f"Training data: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)")
    print(f"Test data: {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} rows)")
    
    return train_df, test_df
