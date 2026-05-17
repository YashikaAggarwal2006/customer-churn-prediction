import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DB_PATH, PLOTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

def load_from_db() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    return df

def run_eda():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_from_db()
    logger.info("Running EDA...")

    # Plot 1 — Churn distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="Exited", data=df, palette="Set2")
    plt.title("Churn Distribution")
    plt.xticks([0,1], ["Stayed","Churned"])
    plt.savefig(PLOTS_DIR / "01_churn_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 01_churn_distribution.png")

    # Plot 2 — Age distribution by churn
    plt.figure(figsize=(8,4))
    sns.histplot(data=df, x="Age", hue="Exited", bins=30, kde=True, palette="Set1")
    plt.title("Age Distribution by Churn")
    plt.savefig(PLOTS_DIR / "02_age_distribution.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 02_age_distribution.png")

    # Plot 3 — Balance by churn
    plt.figure(figsize=(8,4))
    sns.boxplot(x="Exited", y="Balance", data=df, palette="Set2")
    plt.title("Balance by Churn")
    plt.xticks([0,1], ["Stayed","Churned"])
    plt.savefig(PLOTS_DIR / "03_balance_by_churn.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 03_balance_by_churn.png")

    # Plot 4 — Correlation heatmap
    plt.figure(figsize=(10,8))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR / "04_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 04_correlation_heatmap.png")

    # Plot 5 — Churn by Geography
    plt.figure(figsize=(7,4))
    sns.countplot(x="Geography", hue="Exited", data=df, palette="Set2")
    plt.title("Churn by Geography")
    plt.savefig(PLOTS_DIR / "05_churn_by_geography.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 05_churn_by_geography.png")

    # Plot 6 — Churn by Gender
    plt.figure(figsize=(6,4))
    sns.countplot(x="Gender", hue="Exited", data=df, palette="Set1")
    plt.title("Churn by Gender")
    plt.savefig(PLOTS_DIR / "06_churn_by_gender.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 06_churn_by_gender.png")

    # Plot 7 — Churn by Number of Products
    plt.figure(figsize=(6,4))
    sns.countplot(x="NumOfProducts", hue="Exited", data=df, palette="Set2")
    plt.title("Churn by Number of Products")
    plt.savefig(PLOTS_DIR / "07_churn_by_products.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 07_churn_by_products.png")

    logger.info(f"All plots saved to {PLOTS_DIR}")