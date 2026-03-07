# src/app.py

from flask import Flask, request, jsonify
import pandas as pd

# Import model functions
from modeling.predict import (
    load_model,
    preprocess_input,
    predict_failure,
    get_shap_explanation
)

# Create Flask app
app = Flask(__name__)

# Load model once when server starts
model = load_model()


# -------------------------------
# HOME ROUTE (for browser test)
# -------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "FactoryGuard AI API is running",
        "endpoints": {
            "health_check": "/health",
            "predict_failure": "/predict"
        }
    })


# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# -------------------------------
# PREDICTION ENDPOINT
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get JSON input
        data = request.get_json(force=True)

        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400

        # Convert JSON → DataFrame
        df_input = preprocess_input(data)

        # Predict failure probability
        probability = predict_failure(model, df_input)

        # SHAP explanation
        explanation = get_shap_explanation(model, df_input)

        # Create response
        response = {
            "failure_probability": float(probability),
            "failure_risk": "High" if probability > 0.5 else "Low/Medium",
            "shap_explanation": explanation,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        threaded=True
    )