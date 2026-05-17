import sqlite3
import pandas as pd
from src.config import CSV_PATH, DB_PATH, TABLE_NAME, OUTPUTS_DIR
from src.logger import get_logger
from src.eda import run_eda
from src.preprocess import preprocess
from src.train import train
from src.explain import explain

logger = get_logger("main")

def load_csv() -> pd.DataFrame:
    logger.info(f"Loading CSV from {str(CSV_PATH)}")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def save_to_db(df: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"Saved to SQLite: {str(DB_PATH)}")

def run_insights() -> None:
    conn = sqlite3.connect(DB_PATH)

    queries = {
        "overall_churn": """
            SELECT COUNT(*) AS total_customers,
                   SUM(Exited) AS churned,
                   ROUND(AVG(Exited)*100,2) AS churn_rate_pct
            FROM customers""",

        "churn_by_country": """
            SELECT Geography, COUNT(*) AS total,
                   SUM(Exited) AS churned,
                   ROUND(AVG(Exited)*100,2) AS churn_rate_pct
            FROM customers GROUP BY Geography
            ORDER BY churn_rate_pct DESC""",

        "churn_by_age_group": """
            SELECT CASE
                WHEN Age < 30 THEN 'Under 30'
                WHEN Age BETWEEN 30 AND 45 THEN '30-45'
                WHEN Age BETWEEN 46 AND 60 THEN '46-60'
                ELSE 'Over 60' END AS age_group,
                COUNT(*) AS total,
                ROUND(AVG(Exited)*100,2) AS churn_rate_pct
            FROM customers GROUP BY age_group
            ORDER BY churn_rate_pct DESC""",

        "high_value_churned": """
            SELECT CustomerId, Age, Balance, CreditScore, Geography
            FROM customers WHERE Exited=1 AND Balance>100000
            ORDER BY Balance DESC LIMIT 10""",

        "churn_by_products": """
            SELECT NumOfProducts, COUNT(*) AS total,
                   ROUND(AVG(Exited)*100,2) AS churn_rate_pct
            FROM customers GROUP BY NumOfProducts""",

        "churn_by_activity": """
            SELECT IsActiveMember, COUNT(*) AS total,
                   ROUND(AVG(Exited)*100,2) AS churn_rate_pct
            FROM customers GROUP BY IsActiveMember""",

        "churn_by_credit_band": """
            SELECT CASE
                WHEN CreditScore < 500 THEN 'Poor (<500)'
                WHEN CreditScore BETWEEN 500 AND 650 THEN 'Fair (500-650)'
                WHEN CreditScore BETWEEN 651 AND 750 THEN 'Good (651-750)'
                ELSE 'Excellent (>750)' END AS credit_band,
                COUNT(*) AS total,
                ROUND(AVG(Exited)*100,2) AS churn_rate_pct
            FROM customers GROUP BY credit_band
            ORDER BY churn_rate_pct DESC"""
    }

    for name, sql in queries.items():
        result = pd.read_sql_query(sql, conn)
        print(f"\n{'─'*40}")
        print(f"  {name.upper().replace('_', ' ')}")
        print(f"{'─'*40}")
        print(result.to_string(index=False))

    conn.close()

def main():
    logger.info("=" * 50)
    logger.info("PHASE 1 — Data Setup & SQL")
    logger.info("=" * 50)
    df = load_csv()
    save_to_db(df)
    run_insights()
    logger.info("Phase 1 complete ✅")

    logger.info("=" * 50)
    logger.info("PHASE 2 — EDA")
    logger.info("=" * 50)
    run_eda()
    logger.info("Phase 2 complete ✅")

    logger.info("=" * 50)
    logger.info("PHASE 3 — Preprocessing & Training")
    logger.info("=" * 50)
    X, y, scaler, feature_names = preprocess()
    best_model, best_name, X_test, y_test = train(X, y, feature_names)
    logger.info("Phase 3 complete ✅")

    logger.info("=" * 50)
    logger.info("PHASE 4 — SHAP Explainability")
    logger.info("=" * 50)
    explain(best_model, X_test, feature_names)
    logger.info("Phase 4 complete ✅")

if __name__ == "__main__":
    main()