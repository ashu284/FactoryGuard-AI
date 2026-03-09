# FactoryGuard AI – Predictive Maintenance System

**Project 1 – Tabular/IoT Prediction**  
**Data Science Engineering Track – Cohort Zeta**  
**Infotact Solutions – Q4 2025 Capstone**

A production-grade system that predicts machine failure 24 hours in advance using sensor data (temperature, torque, rotational speed, tool wear, etc.), with high performance, SHAP explainability, and real-time Flask API.

**Dataset**  
AI4I 2020 Predictive Maintenance Dataset (10,000 rows, ~3.4% failure rate)

**Tech Stack** (aligned with memo)  
- Python 3.10+  
- Pandas, NumPy, Scikit-learn  
- XGBoost (final model)  
- SMOTE (imbalance)  
- SHAP (explainability)  
- Flask (API)  
- Joblib (serialization)

**Project Structure**
FACTORYGUARD-AI/
├── data/
│   ├── raw/ai4i_predictive_maintenance.csv
│   └── processed/
├── models/
│   ├── best_model_xgb.pkl
│   └── feature_names_clean.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_explainability_SHAP.ipynb
├── src/
│   ├── modeling/
│   │   └── predict.py          # prediction + SHAP logic
│   └── app.py                  # Flask API
├── reports/
│   ├── shap_summary_beeswarm_week3.png
│   ├── shap_summary_bar_week3.png
│   └── confusion_matrix_xgb.png
├── requirements.txt
└── README.md

## 4-Week Sprint Summary

### Week 1 – Data Engineering & EDA
- Loaded raw CSV
- Dropped non-predictive columns: UDI, Product ID, TWF, HDF, PWF, OSF, RNF
- One-hot encoded `Type` → `Type_L`, `Type_M`
- Added temporal features:
  - Rolling mean (4h ≈ 240 min window)
  - EMA (alpha=0.1)
  - Lag (1h ≈ 60 min shift)
- Filled missing values (bfill + 0 fallback)
- Class distribution: ~96.61% no failure, ~3.39% failure

**Key files**  
- `notebooks/01_eda.ipynb`  
- Processed data saved to `data/processed/`

### Week 2 – Modeling & Baseline
- Stratified train-test split (80/20)
- SMOTE oversampling on training set only
- Trained XGBoost with RandomizedSearchCV (optimized for F1-score)
- Final model: `best_model_xgb.pkl`
- Evaluation: High recall on failure class, good macro F1
- Saved model + cleaned feature names
| Metric | Value |
|--------|-------|
| F1-Score (failure class) | 0.85 |
| Recall (failure class) | 0.92 |
| Precision | 0.78 |
| Latency (API) | <100 ms |

**Key files**  
- `notebooks/02_baseline_model.ipynb`  
- `notebooks/03_feature_engineering.ipynb`

### Week 3 – Interpretability & Trust (XAI)
- Used SHAP TreeExplainer on XGBoost
- Computed SHAP values on test set
- Generated & saved:
  - Beeswarm summary plot → feature impact direction
  - Bar summary plot → importance ranking
  - Force plots for multiple failure cases
- Validation:
  - Top features: Tool_wear, torque_roll_4h, Torque, tool_wear_ema
  - High tool wear/torque/temperature → strong positive SHAP (failure risk up)
  - Aligns with manufacturing physics: wear & stress drive failures
  <image-card alt="SHAP Beeswarm" src="reports/shap_summary_beeswarm_week3.png" ></image-card>
<image-card alt="API Response" src="reports/api_response.png" ></image-card>

**Key files**  
- `notebooks/04_explainability_SHAP.ipynb`  
- Outputs: `reports/shap_summary_beeswarm_week3.png`, `reports/shap_summary_bar_week3.png`, `reports/force_plot_failure_*.png`

### Week 4 – Deployment Wrapper (Flask API)
- Modular inference in `src/modeling/predict.py`:
  - Load model
  - Preprocess input (clean names, fill missing with 0)
  - Predict probability
  - Compute SHAP + top contributors
- Flask server `src/app.py`:
  - `/` → home page with endpoints
  - `/health` → model status
  - `/predict` (POST) → probability + SHAP explanation

**How to run the API**

1. Install dependencies
 pip install flask joblib shap pandas numpy xgboost requests
2. Start server
cd src
python app.py

→ See: `* Running on http://127.0.0.1:5000`

Keep this terminal open.

3. Test health (in new PowerShell)
curl.exe http://127.0.0.1:5000/health

Expected:

```json
{"model_loaded": true, "status": "healthy"}
4. Test prediction (easiest way)
python -c "import requests; r = requests.post('http://127.0.0.1:5000/predict', json={'Air_temperature':298.1, 'Process_temperature':308.6, 'Rotational_speed':1551, 'Torque':42.8, 'Tool_wear':0, 'Type_L':0, 'Type_M':0}); print(r.text)"
Expected output (example):
{
  "failure_probability": 3.256e-07,
  "failure_risk": "Low/Medium",
  "shap_explanation": {
    "expected_value": -0.00625,
    "top_contributors": [
      {"feature": "Torque", "shap_value": 3.071, "feature_value": 42.8},
      {"feature": "Tool_wear", "shap_value": 2.263, "feature_value": 0.0},
      ...
    ]
  }
}