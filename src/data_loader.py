import pandas as pd
import os
from src.config import RAW_DATA_FILE

def load_data(file_path=None):
    """
    Load the raw sales data from CSV.
    
    Args:
        file_path (str): Path to the CSV file. Defaults to RAW_DATA_FILE from config.
        
    Returns:
        pd.DataFrame: Loaded dataframe with parsed dates.
    """
    if file_path is None:
        file_path = RAW_DATA_FILE
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please generate data first.")
        
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date, store, product
    df = df.sort_values(['date', 'store_id', 'product_id']).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Test loading
    try:
        df = load_data()
        print("Data loaded successfully.")
        print(df.info())
    except Exception as e:
        print(f"Error: {e}")
