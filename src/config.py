from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT_DIR / "data" / "raw"
DATA_PROC   = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR   = ROOT_DIR / "outputs" / "plots"

CSV_PATH    = DATA_RAW    / "churn.csv"
DB_PATH     = DATA_PROC   / "churn.db"
MODEL_PATH  = OUTPUTS_DIR / "model.pkl"
TABLE_NAME  = "customers"
TARGET_COL  = "Exited"
DROP_COLS   = ["RowNumber", "CustomerId", "Surname"]