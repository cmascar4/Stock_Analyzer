from pathlib import Path

TICKERS = [
    "SPY", "VOO", "VTI", "QQQ", "QQQM", "VXUS",
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN",
    "META", "TSLA", "AVGO", "AMD", "NFLX", "CRM",
    "ADBE", "ORCL", "INTC", "JPM", "PLTR", "COIN",
    "UBER", "PYPL", "FZROX", "FBGRX", "TSM", "HOOD",
    "SCHD", "SNDK", "GLD",
]
# Deduplicate while preserving order
TICKERS = list(dict.fromkeys(TICKERS))

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Data settings
TRAINING_PERIOD = "5y"

# Model hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_DEPTH = 5
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

# Dashboard
DEFAULT_PERIOD = "2y"
