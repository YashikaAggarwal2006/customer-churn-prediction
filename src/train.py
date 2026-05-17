import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from src.config import MODEL_PATH, PLOTS_DIR, OUTPUTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

def train(X, y, feature_names):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(eval_metric="logloss", random_state=42)
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    logger.info("Training models...")
    for name, model in models.items():
        # Cross validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)

        results[name] = {"model": model, "auc": auc}

        logger.info(f"  {name}: AUC={auc:.4f} | CV AUC={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n{'─'*40}\n{name}\n{'─'*40}")
        print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    # Pick best model
    best_name  = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    logger.info(f"Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

    # ROC curve
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba_best, ax=ax, name=best_name)
    plt.title(f"ROC Curve — {best_name}")
    plt.savefig(PLOTS_DIR / "08_roc_curve.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 08_roc_curve.png")

    # Confusion matrix
    y_pred_best = best_model.predict(X_test)
    cm   = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.savefig(PLOTS_DIR / "09_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 09_confusion_matrix.png")

    # Save best model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)
    logger.info(f"Model saved to {str(MODEL_PATH)}")

    return best_model, best_name, X_test, y_test