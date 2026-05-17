import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st
from src.config import MODEL_PATH, PLOTS_DIR

# ── Load model ───────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)
model      = saved["model"]
model_name = saved["name"]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Predictor")
st.markdown(f"**Model:** {model_name}")
st.markdown("---")

# ── Sidebar inputs ───────────────────────────────────────────
st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.slider("Credit Score",      300, 850, 650)
age          = st.sidebar.slider("Age",                18,  92,  35)
tenure       = st.sidebar.slider("Tenure (years)",      0,  10,   5)
balance      = st.sidebar.number_input("Balance",       0, 300000, 50000)
num_products = st.sidebar.selectbox("Num of Products", [1, 2, 3, 4])
has_card     = st.sidebar.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
is_active    = st.sidebar.selectbox("Is Active Member",[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
salary       = st.sidebar.number_input("Estimated Salary", 0, 200000, 60000)
geography    = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender       = st.sidebar.selectbox("Gender", ["Male", "Female"])

# ── Encode inputs ────────────────────────────────────────────
geo_map    = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}

input_data = np.array([[
    credit_score,
    geo_map[geography],
    gender_map[gender],
    age, tenure, balance,
    num_products, has_card,
    is_active, salary
]])

feature_names = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# ── Predict ──────────────────────────────────────────────────
proba = model.predict_proba(input_data)[0][1]
risk  = "🔴 HIGH" if proba > 0.7 else "🟡 MEDIUM" if proba > 0.4 else "🟢 LOW"

col1, col2, col3 = st.columns(3)
col1.metric("Churn Probability", f"{proba*100:.1f}%")
col2.metric("Risk Level", risk)
col3.metric("Model", model_name)

st.progress(float(proba))
st.markdown("---")

# ── SHAP explanation ─────────────────────────────────────────
st.subheader("Why is this customer at risk?")
explainer   = shap.Explainer(model, input_data)
shap_values = explainer(input_data)

fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── SHAP summary from saved plot ─────────────────────────────
st.subheader("📊 Top Features Driving Churn")
st.image(
    str(PLOTS_DIR / "10_shap_summary.png"),
    caption="SHAP Feature Importance",
    use_column_width=True
)