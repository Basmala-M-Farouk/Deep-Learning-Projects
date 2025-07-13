# 👐 GestureNet Project: Real-time Gesture Recognition with Deep Learning

## ✨ Overview

This repository hosts the `GestureNet Project` Jupyter Notebook, a comprehensive guide to building and evaluating a deep learning model for real-time gesture recognition. It leverages Convolutional Neural Networks (CNNs) to accurately interpret hand gestures from video input, providing a foundational understanding of gesture recognition systems. The project includes data preprocessing, model architecture design, training, evaluation, and visualization of results.

## 🚀 Features

* **Data Preprocessing**: Utilizes `MediaPipe` for efficient hand landmark extraction, preparing data for model training.
* **Deep Learning Models**: Implements both a standard Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for gesture classification, demonstrating their comparative performance.
* **Optimizer Comparison**: Explores the impact of various optimizers (SGD, SGD with Momentum, Adagrad, RMSProp, Adam) on model training and performance for both ANN and CNN architectures.
* **Comprehensive Evaluation**: Provides detailed metrics including training loss, training accuracy, validation loss, validation accuracy, test loss, and test accuracy for each model and optimizer combination.
* **Visualizations**: Generates insightful plots for loss and accuracy over epochs, and confusion matrices to visually assess model performance and identify misclassifications.

## 📁 Project Structure

The core of this project is the `GestureNet Project.ipynb` Jupyter Notebook, which contains:

* **Data Loading and Preparation**: Steps to load raw video data and convert it into a suitable format for neural network input using MediaPipe.
* **Model Definition**: Python code for constructing the ANN and CNN architectures using TensorFlow/Keras.
* **Training Loops**: Implementation of training procedures for various optimizers.
* **Evaluation and Reporting**: Code to evaluate the trained models and present performance tables and graphs.

## 🛠️ Getting Started

### Data Source

The dataset used for training and evaluating the gesture recognition models is the [Rock-Paper-Scissors Dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors) from TensorFlow Datasets.

### Prerequisites

To run this notebook, you will need:

* Python 3.x
* Jupyter Notebook or JupyterLab
* Key Python libraries:
    * `tensorflow` (or `keras`)
    * `numpy`
    * `pandas`
    * `matplotlib`
    * `seaborn`
    * `scikit-learn`
    * `mediapipe`

You can install the required libraries using pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn mediapipe
