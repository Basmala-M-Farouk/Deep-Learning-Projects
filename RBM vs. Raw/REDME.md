# 👗💡RBM Features: Unlocking Insights in Fashion MNIST 

## Overview

This repository explores the application of **Restricted Boltzmann Machines (RBMs)** as feature extractors to enhance the performance of a Multi-Layer Perceptron (MLP) on the widely-used **Fashion MNIST dataset**. The project delves into the process of unsupervised pre-training with RBMs and compares the classification accuracy of an MLP when fed with RBM-generated features versus raw image data. It provides a clear demonstration of setting up, training, and evaluating these hybrid models. 🧠✨

## Features

* **Fashion MNIST Dataset:** Utilizes the [Fashion MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist), a common benchmark for image classification, featuring 10 categories of clothing items. 👕👖
* **Comprehensive Data Preparation:**
    * Loads and reshapes 28x28 pixel grayscale images into 1D vectors.
    * Scales pixel values to the [0, 1] range for neural network compatibility.
    * Binarizes the data, which is a common requirement for Bernoulli RBMs. ✂️
* **Restricted Boltzmann Machine (RBM):** Implements `BernoulliRBM` from `scikit-learn` for unsupervised pre-training and feature extraction. 🔄
* **Multi-Layer Perceptron (MLP):** Develops two distinct MLP models using TensorFlow/Keras:
    * One trained directly on the raw, preprocessed Fashion MNIST images.
    * Another trained on the features extracted by the RBM. 🚀
* **Pipeline Integration:** Uses `scikit-learn`'s `Pipeline` to streamline the process of RBM feature extraction followed by MLP classification. ⛓️
* **Performance Comparison:** Directly compares the `Test Loss` and `Test Accuracy` of both MLP configurations (with RBM features vs. raw images), providing clear insights into the impact of RBM pre-training. 📊
* **TensorFlow & Keras:** Built with the latest TensorFlow 2.x and Keras API for efficient deep learning development. 🐍

## Project Structure

* `RBM.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, RBM training, MLP construction, comparative training, and evaluation. 📝

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow 2.x
* NumPy
* scikit-learn
* `tabulate` (for printing results table)

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Basmala-M-Farouk/Deep-Learning-Projects.git](https://github.com/Basmala-M-Farouk/Deep-Learning-Projects.git) # Replace with your actual repo URL
    cd Deep-Learning-Projects/RBM vs. Raw/REDME.md # Adjust path if RBM.ipynb is in a subfolder
    ```
2.  Install the required libraries:
    ```bash
    pip install tensorflow numpy scikit-learn tabulate
    ```

### Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook RBM.ipynb
    ```
2.  Run all cells in the notebook. This will:
    * Load and preprocess the Fashion MNIST dataset.
    * Train the Bernoulli RBM.
    * Train two separate MLPs (one with RBM features, one with raw data).
    * Display a comparative table of their test performance. ▶️

## Results

The project provides a direct comparison of MLP performance with and without RBM-extracted features. The analysis indicates that, in this specific experimental setup, training the MLP directly on the raw image data yielded better results (lower loss, higher accuracy) than using features pre-processed by the RBM. This suggests that the RBM features did not provide a significant performance boost in this particular classification task.

| Model                 | Test Loss | Test Accuracy |
|:----------------------|:----------|:--------------|
| MLP with RBM Features | 0.4354    | 84.76%        |
| MLP with Raw Images   | 0.3432    | 88.98%        |


*(Note: The exact values might vary slightly based on training randomness.)*

## Contributing

Contributions are always welcome! If you have ideas for improving RBM integration, exploring different architectures, or optimizing the pipeline, please fork this repository, open an issue, or submit a pull request. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/ 
