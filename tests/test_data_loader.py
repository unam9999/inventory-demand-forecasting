import pytest
import pandas as pd
import os
from src.data_loader import load_data
from src.config import RAW_DATA_FILE

def test_load_data_exists():
    """Test that data file exists and can be loaded."""
    assert os.path.exists(RAW_DATA_FILE), "Raw data file does not exist."
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_data_columns():
    """Test that loaded data has expected columns."""
    df = load_data()
    expected_columns = [
        'date', 'store_id', 'product_id', 'on_hand_inventory', 'units_sold',
        'price', 'promotion', 'holiday', 'day_of_week', 'lead_time_days', 'stockout_flag'
    ]
    for col in expected_columns:
        assert col in df.columns, f"Column {col} missing from data."

def test_date_parsing():
    """Test that date column is parsed as datetime."""
    df = load_data()
    assert pd.api.types.is_datetime64_any_dtype(df['date']), "Date column not parsed as datetime."
