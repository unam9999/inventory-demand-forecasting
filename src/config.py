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

# Logging configuration
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Parameters
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
TEST_SIZE_DAYS = int(os.environ.get("TEST_SIZE_DAYS", 60))  # Last 60 days for testing

# Model Hyperparameters (Default)
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", 100))
RF_MAX_DEPTH = os.environ.get("RF_MAX_DEPTH", None)
if RF_MAX_DEPTH is not None:
    RF_MAX_DEPTH = int(RF_MAX_DEPTH)

# Feature Engineering
USE_LAG_FEATURES = os.environ.get("USE_LAG_FEATURES", "True") == "True"
USE_ROLLING_STATS = os.environ.get("USE_ROLLING_STATS", "True") == "True"
