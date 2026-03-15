import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'historical_sales_sample.csv')
TRAIN_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'training_data.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_FILE = os.path.join(MODELS_DIR, 'best_model.pkl')
SCALER_FILE = os.path.join(MODELS_DIR, 'scaler.pkl')

# Reports paths
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Parameters
RANDOM_SEED = 42
TEST_SIZE_DAYS = 60  # Last 60 days for testing
