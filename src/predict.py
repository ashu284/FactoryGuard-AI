import pickle
import pandas as pd

# Load model
with open("../models/factoryguard_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example new machine data
new_data = pd.DataFrame({
    "Air temperature [K]": [300],
    "Process temperature [K]": [310],
    "Rotational speed [rpm]": [1500],
    "Torque [Nm]": [40],
    "Tool wear [min]": [120]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("⚠ Machine Failure Predicted in Next 24 Hours!")
else:
    print("✅ Machine Operating Normally")