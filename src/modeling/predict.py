# src/modeling/predict.py

import joblib
import shap
import pandas as pd
import numpy as np

# These are the EXACT cleaned feature names from your feature_names_clean.csv
# Copy-pasted from your screenshot – no changes
EXPECTED_FEATURES = [
    'Air_temperature',
    'Process_temperature',
    'Rotational_speed',
    'Torque',
    'Tool_wear',
    'Type_L',
    'Type_M',
    'tool_wear_roll_4h',
    'torque_roll_4h',
    'rpm_roll_4h',
    'tool_wear_ema',
    'torque_ema',
    'tool_wear_lag_1h'
]

def load_model(model_path='../models/best_model_xgb.pkl'):
    """Load the trained XGBoost model."""
    return joblib.load(model_path)

def preprocess_input(data_dict):
    """
    Convert incoming JSON dict to a Pandas DataFrame with EXACTLY
    the same columns and order as during training.
    Missing features get filled with 0 (simple default for testing).
    """
    df = pd.DataFrame([data_dict])

    # Add any missing columns with 0
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # Force the exact column order the model expects
    df = df[EXPECTED_FEATURES]

    return df

def predict_failure(model, df_input):
    """Return failure probability (class 1 probability)."""
    prob = model.predict_proba(df_input)[:, 1][0]
    return float(prob)

def get_shap_explanation(model, df_input):
    """Compute SHAP values and return a dict ready for JSON response."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # For single row (we have only 1 row)
    explanation = {
        "expected_value": float(explainer.expected_value),
        "shap_values": shap_values[0].tolist(),
        "features": df_input.iloc[0].to_dict(),
        "top_contributors": []
    }

    # Add top 5 contributors by absolute SHAP value (useful for users)
    contrib = pd.Series(shap_values[0], index=EXPECTED_FEATURES)
    top = contrib.abs().sort_values(ascending=False).head(5)
    for feature, value in top.items():
        explanation["top_contributors"].append({
            "feature": feature,
            "shap_value": float(value),
            "feature_value": float(df_input[feature].iloc[0])
        })

    return explanation

# Quick local test (run: python predict.py in src folder)
if __name__ == "__main__":
    print("Testing predict.py functions...")
    model = load_model()
    
    # Minimal test input (matches your cleaned names)
    test_input = {
        "Air_temperature": 298.1,
        "Process_temperature": 308.6,
        "Rotational_speed": 1551,
        "Torque": 42.8,
        "Tool_wear": 0,
        "Type_L": 0,
        "Type_M": 0
        # Rolling/EMA/lag will be filled with 0 automatically
    }
    
    df_test = preprocess_input(test_input)
    prob = predict_failure(model, df_test)
    expl = get_shap_explanation(model, df_test)
    
    print(f"Failure Probability: {prob:.4f}")
    print("SHAP Top Contributors:", expl["top_contributors"])