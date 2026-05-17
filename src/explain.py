import shap
import matplotlib.pyplot as plt
from src.config import PLOTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

def explain(model, X_test, feature_names):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating SHAP explanations...")

    # Create explainer
    explainer   = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Plot 1 — SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_shap_summary.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 10_shap_summary.png")

    # Plot 2 — SHAP bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_shap_bar.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 11_shap_bar.png")

    # Plot 3 — Waterfall plot for single prediction
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_shap_waterfall.png", bbox_inches="tight")
    plt.close()
    logger.info("  ✓ 12_shap_waterfall.png")

    logger.info("SHAP explanations complete ✅")