from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "customers.csv"
DATA_SCORED = ROOT / "data" / "processed" / "customers_scored.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
N_CUSTOMERS = 25_430
TARGET_CHURN_RATE = 0.148
N_CLUSTERS = 4
TEST_SIZE = 0.20

# Risk bands applied to the calibrated churn probability
RISK_THRESHOLDS = (0.30, 0.60)   # <0.30 Low | 0.30-0.60 Medium | >=0.60 High
PREDICTION_THRESHOLD = 0.50      # churn prediction = YES if probability >= this

for _d in (DATA_RAW.parent, DATA_SCORED.parent, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
