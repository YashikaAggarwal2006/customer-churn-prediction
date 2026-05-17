import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from src.config import DB_PATH, DROP_COLS, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)

def preprocess():
    # Load from DB
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    logger.info(f"Loaded {len(df):,} rows for preprocessing")

    # Drop useless columns
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df["Gender"]    = le.fit_transform(df["Gender"])
    df["Geography"] = le.fit_transform(df["Geography"])

    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"Before SMOTE: {y.value_counts().to_dict()}")

    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    logger.info("Preprocessing complete ✅")
    return X_scaled, y_res, scaler, X.columns.tolist()