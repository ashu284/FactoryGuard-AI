# 🏭 FactoryGuard AI – IoT Predictive Maintenance Engine

## 📌 Introduction

FactoryGuard AI is an Industrial AI system designed to predict machine failure before it happens.  
The goal of this project is to reduce unplanned downtime in manufacturing environments using Machine Learning.

In large-scale industries, unexpected machine failure can cost approximately $10,000 per hour.  
This system predicts failure 24 hours in advance using sensor data, helping companies save money and improve operational efficiency.

---

## 🎯 Problem Statement

A manufacturing facility operates 500+ robotic machines.

Challenges:
- Unexpected equipment breakdown
- High downtime cost
- Manual maintenance scheduling
- Lack of predictive monitoring

Objective:
Build a Binary Classification Model that predicts whether a machine will fail within the next 24 hours based on sensor readings.

---

## 📊 Dataset Description

Dataset: AI4I Predictive Maintenance Dataset

The dataset contains sensor readings collected from industrial machines.

### Input Features:
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]
- Machine Type (Categorical)

### Target Variable:
- Machine Failure  
  - 0 → No Failure  
  - 1 → Failure  

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP (Model Explainability)
- Matplotlib
- Jupyter Notebook

---

## 📈 Project Workflow

### 1️⃣ Data Preprocessing
- Removed unnecessary columns (UDI, Product ID)
- Converted categorical variables using encoding
- Handled missing values
- Created clean processed dataset

### 2️⃣ Exploratory Data Analysis (EDA)
- Checked distribution of features
- Visualized correlations
- Identified important sensor variables

### 3️⃣ Model Training
- Used Random Forest Classifier
- Split data into training and testing sets
- Trained model on processed dataset

### 4️⃣ Model Evaluation
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 5️⃣ Model Explainability
- Used SHAP (SHapley Additive exPlanations)
- Identified most important features affecting prediction
- Ensured transparency and interpretability

### 6️⃣ Model Saving
- Saved trained model as `.pkl` file
- Created prediction script for real-time use

---



This ensures that the AI model decisions are transparent and trustworthy.

---

## ⚙ How To Run This Project

### Step 1: Clone Repository
