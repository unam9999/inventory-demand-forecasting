# API Documentation

## `src.data_loader`

### `load_data(file_path=None)`
Loads raw sales data from CSV.
- **Args**: `file_path` (str, optional): Path to CSV.
- **Returns**: `pd.DataFrame`: Dataframe with parsed dates.

## `src.preprocessing`

### `preprocess_data(df)`
Cleans raw data.
- **Args**: `df` (pd.DataFrame): Raw data.
- **Returns**: `pd.DataFrame`: Cleaned data.

### `split_data(df, test_days=60)`
Splits data into train and test sets by date.
- **Args**: 
    - `df` (pd.DataFrame): Data.
    - `test_days` (int): Number of days for test set.
- **Returns**: `tuple`: (train_df, test_df)

## `src.feature_engineering`

### `create_features(df)`
Generates time-series features.
- **Args**: `df` (pd.DataFrame): Input data.
- **Returns**: `pd.DataFrame`: Data with new features (lags, rolling means).

### `encode_categorical(df, categorical_cols=None)`
One-hot encodes categorical columns.
- **Args**: 
    - `df` (pd.DataFrame): Input data.
    - `categorical_cols` (list): Columns to encode.
- **Returns**: `pd.DataFrame`: Encoded data.

## `src.models`

### `get_random_forest_model()`
Returns a configured Random Forest Regressor.
- **Returns**: `sklearn.ensemble.RandomForestRegressor`
