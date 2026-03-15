import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data, split_data
from src.feature_engineering import create_features

@pytest.fixture
def sample_df():
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = {
        'date': dates,
        'store_id': [1]*100,
        'product_id': [101]*100,
        'units_sold': np.random.randint(0, 20, 100),
        'price': [10.0]*100,
        'on_hand_inventory': [50]*100
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_df):
    # Introduce some bad data
    sample_df.loc[0, 'units_sold'] = -1
    sample_df.loc[1, 'price'] = np.nan
    
    processed = preprocess_data(sample_df)
    
    assert len(processed) < len(sample_df) # Should drop negative sales
    assert processed['price'].isna().sum() == 0 # Should fill NaNs

def test_split_data(sample_df):
    train, test = split_data(sample_df, test_days=10)
    assert len(test) == 10
    assert len(train) == 90
    assert train['date'].max() < test['date'].min()

def test_create_features(sample_df):
    # Need enough data for lags (30 days max lag/rolling)
    df = create_features(sample_df)
    
    # Check if columns exist
    expected_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30']
    for col in expected_cols:
        assert col in df.columns
        
    # Check that we dropped NaNs
    assert df.isna().sum().sum() == 0
