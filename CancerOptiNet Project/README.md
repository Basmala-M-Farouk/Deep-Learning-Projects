# 🧠 CancerOptiNet  
**Deep Learning Optimization for Cancer Survival Prediction**

## 📘 Project Description

This project applies deep learning techniques to a simulated global cancer patient dataset (2015–2024) to predict key outcomes such as **survival years**, **treatment cost**, and **severity score**. The main goal is to evaluate and compare the performance of various gradient-based optimizers (SGD, Adam, RMSprop, Adagrad) in terms of convergence speed, accuracy, and overall model performance.

---

## 🩺 Dataset Overview

The dataset includes global cancer data with the following features:

- **Demographics:** Age, Gender, Country/Region  
- **Medical Info:** Cancer Type, Stage, Severity Score  
- **Lifestyle & Risk Factors:** Smoking, Alcohol, Environmental Exposure  
- **Target Variables:**  
  - 🎯 Survival Years  
  - 💰 Treatment Cost  
  - 🚨 Severity Score  

---

## 🔧 Tasks Breakdown

### 1️⃣ Data Preprocessing
- Normalized numeric features
- One-hot encoded categorical features
- Train-test split (80% training, 20% testing)

### 2️⃣ Optimizer Comparison
Train the same neural network using different optimizers:
- 🔹 **Stochastic Gradient Descent (SGD)**
- 🔹 **Adam**
- 🔹 **RMSprop**
- 🔹 **Adagrad**

Each optimizer is evaluated for:
- Convergence behavior
- Learning rate sensitivity
- Stability during training

### 3️⃣ Evaluation & Visualization
- Track and plot:
  - Training vs. Validation Loss
  - MAE / MSE (for regression)
  - Accuracy (if applicable)
- Final performance metrics per optimizer
- Discuss: Which optimizer performed best and why?

---

## 🧠 Learning Outcomes

### ✅ See Gradient Descent in Action
- Observe how each optimizer updates weights
- Compare their performance across metrics

### ✅ Understand Learning Rate & Momentum
- Explore the effects of learning rate tuning
- See how momentum and adaptive learning impact results

### ✅ Learn to Evaluate Optimizer Quality
- Spot overfitting, slow convergence, and poor generalization
- Understand optimizer choice in real-world problems

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib & Seaborn
- Google Colab

---

## 📊 Results & Insights

> After training and comparing all models, the project concludes with a detailed analysis of each optimizer’s behavior, convergence pattern, and generalization ability on real-world health data.

---

