# Optimizer Comparison for Cancer Patient Outcome Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project compares different optimization algorithms for training a neural network on global cancer patient data (2015-2024) to predict treatment outcomes.

## 📌 Project Overview

The goal is to evaluate how different optimizers affect model performance when predicting:
- Treatment cost (USD)
- Survival years
- Severity score

Key features include:
- Patient demographics (age, gender, country)
- Cancer characteristics (type, stage)
- Risk factors (genetics, pollution, lifestyle)

## 🛠️ Technical Implementation

### Data Preprocessing
- Numeric features normalized using `MinMaxScaler`
- Categorical features one-hot encoded
- Dataset split 80% train / 20% test

### Model Architecture
```python
Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(3, activation='linear')  # Multi-output regression
])
