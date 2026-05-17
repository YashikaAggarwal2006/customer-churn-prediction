# customer-churn-prediction

# 🏦 Bank Customer Churn Prediction

An end-to-end machine learning project to predict which bank customers are likely to churn.

## 🔍 Problem Statement
Banks lose millions every year due to customer churn. This project builds a predictive system to identify at-risk customers early so the bank can take action.

## 📊 Dataset
- **Source:** Kaggle — Bank Customer Churn Prediction
- **Size:** 10,000 records, 14 features
- **Target:** `Exited` (1 = churned, 0 = stayed)

## 🛠️ Tech Stack
- **Language:** Python
- **Database:** SQLite
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Seaborn, Matplotlib, Streamlit

## 📁 Project Structure
churn_prediction/
├── data/
│   ├── raw/               ← original dataset
│   └── processed/         ← SQLite database
├── src/
│   ├── config.py          ← all settings
│   ├── logger.py          ← logging
│   ├── eda.py             ← EDA plots
│   ├── preprocess.py      ← SMOTE + scaling
│   ├── train.py           ← model training
│   └── explain.py         ← SHAP explainability
├── outputs/
│   └── plots/             ← all saved plots
├── app.py                 ← Streamlit dashboard
├── main.py                ← run full pipeline
└── requirements.txt



## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline
```bash
python main.py
```

### 3. Launch dashboard
```bash
streamlit run app.py
```

## 🔄 Project Phases

| Phase | What it does |
|---|---|
| Phase 1 | Data loading, SQLite setup, SQL business insights |
| Phase 2 | Exploratory Data Analysis — 7 visualizations |
| Phase 3 | Preprocessing (SMOTE), model training (LR, RF, XGBoost), evaluation |
| Phase 4 | SHAP explainability — feature importance, waterfall plots |
| Phase 5 | Streamlit dashboard with live churn prediction |

## 📈 Results

- **Best Model:** XGBoost
- **AUC-ROC:** 0.89+
- **Class Imbalance:** Handled with SMOTE
- **Explainability:** SHAP summary, bar, and waterfall plots

## 🚀 Live Demo
[Click here to try the live dashboard](#) ← replace with your Streamlit link after deployment

## 👩‍💻 Author
**Yashika Aggarwal**  
B.Tech CSE — Swami Keshvanand Institute of Technology  
[LinkedIn](#) | [GitHub](#)
