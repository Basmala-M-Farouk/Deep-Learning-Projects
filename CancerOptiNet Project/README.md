# 🧬📊 CancerOptiNet: AI-Driven Cancer Outcome Prediction 

## Overview

This repository presents **CancerOptiNet**, a deep learning project focused on predicting critical cancer outcomes: `Treatment Cost (USD)`, `Survival Years`, and `Target Severity Score`. Utilizing a comprehensive dataset of global cancer patients from 2015-2024, this project demonstrates a robust pipeline for data preprocessing, deep learning model development, and performance evaluation. Our goal is to leverage AI to gain insights into cancer prognosis and treatment. 💡

## Features

* **Global Cancer Dataset (2015-2024):** Processes a real-world dataset of cancer patients, including various demographic and medical factors. 🌍
* **Multi-Target Regression:** Predicts three key continuous outcomes simultaneously: treatment cost, survival years, and severity score. 🎯
* **Advanced Data Preprocessing:** Implements `MinMaxScaler` for numerical features and `OneHotEncoder` for categorical variables, streamlined with `ColumnTransformer` and `Pipeline`. ✂️
* **Deep Learning Model (Keras):** Utilizes a multi-layer Sequential neural network with `Dense` and `Dropout` layers, incorporating `L2 regularization` to prevent overfitting. 🧠
* **Optimizer Comparison:** Evaluates model performance across various optimizers including `SGD`, `Adam`, `RMSprop`, and `Adagrad` to identify the most effective training strategy. ⚙️
* **Comprehensive Evaluation Metrics:** Models are compiled with Mean Squared Error (MSE) as the loss function and Mean Absolute Error (MAE) as a key evaluation metric. ✅
* **Google Drive Integration:** Designed to seamlessly access datasets and save models from Google Drive in a Colab environment. 📂
* **Python & Libraries:** Built using Python with essential libraries like `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `TensorFlow/Keras`. 🐍

## Project Structure

* `CancerOptiNet_Project.ipynb`: The main Jupyter Notebook containing the complete code for data handling, model building, training, and evaluation. 📝

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow 2.x
* pandas
* numpy
* matplotlib
* scikit-learn

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/CancerOptiNet.git](https://github.com/YourUsername/CancerOptiNet.git)
    cd CancerOptiNet
    ```
2.  Install the required libraries:
    ```bash
    pip install tensorflow pandas numpy matplotlib scikit-learn
    ```

### Usage

1.  Ensure your dataset (`global_cancer_patients_2015_2024.csv`) is accessible via Google Drive at `/content/drive/MyDrive/opt data/`.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook CancerOptiNet_Project.ipynb
    ```
3.  Run all cells in the notebook to execute the full pipeline, from data loading and preprocessing to model training and evaluation. ▶️

## Results

The project meticulously trains models with different optimizers and records their performance in terms of loss (MSE) and MAE on training, validation, and test datasets. The results table summarizes the effectiveness of each optimizer. 📈

*(Please refer to the `CancerOptiNet_Project.ipynb` notebook for the detailed results table and performance plots generated during execution, as training outputs can vary slightly and the complete table is generated dynamically.)*

## Contributing

We welcome contributions to enhance CancerOptiNet! Feel free to fork the repository, open issues for bugs or feature requests, and submit pull requests. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/
