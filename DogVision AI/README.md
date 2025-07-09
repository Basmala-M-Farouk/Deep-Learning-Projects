# 🐾🧠DogVision AI: InceptionV3 Canine Classifier 

## Overview

This repository showcases a deep learning project focused on **dog breed classification** using the **Stanford Dogs Dataset**. We explore the effectiveness of transfer learning with the powerful **InceptionV3** model and analyze the impact of **data augmentation** on model performance. This project provides a comprehensive walkthrough of building, training, and evaluating convolutional neural networks for image classification tasks. 🐶✨

## Features

* **Stanford Dogs Dataset Integration:** Seamless loading and preparation of the Stanford Dogs Dataset. 📊
* **InceptionV3 Transfer Learning:** Leverages a pre-trained InceptionV3 model for robust feature extraction. 🚀
* **Data Augmentation:** Implements image augmentation techniques to improve model generalization and reduce overfitting. 📈
* **Dual Model Training:** Trains and compares two models: one with data augmentation and one without, highlighting the benefits of augmentation. 🔄
* **Comprehensive Evaluation:** Detailed evaluation of model performance on training, validation, and test sets, including loss and accuracy metrics. ✅
* **Performance Visualization:** Generates plots to visualize training and validation metrics, offering clear insights into model learning curves. 📉📈
* **TensorFlow & Keras:** Built with the latest TensorFlow 2.x and Keras API for efficient deep learning development. 🐍

## Project Structure

* `Assignment2_DeepLearning.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model building, training, evaluation, and visualization. 📝

## Getting Started

### Prerequisites

* Python 3.x 🐍
* TensorFlow 2.x
* TensorFlow Datasets
* Matplotlib
* Pandas

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/DogBreedPro.git](https://github.com/YourUsername/DogBreedPro.git)
    cd DogBreedPro
    ```
2.  Install the required libraries:
    ```bash
    pip install tensorflow tensorflow-datasets matplotlib pandas
    ```

### Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Assignment2_DeepLearning.ipynb
    ```
2.  Run all cells in the notebook to execute the entire project pipeline, from data loading to model evaluation and visualization. ▶️

## Results

The project provides a comparative analysis of models trained with and without data augmentation. Final results and performance metrics are clearly presented within the notebook, demonstrating the impact of augmentation on improving test accuracy. 🏆

| Model             | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy |
|:------------------|:-----------|:---------------|:----------------|:--------------------|:----------|:--------------|
| Pretrained\_no\_aug | 0.138355   | 0.969314       | 0.342636        | 0.881667            | 0.331963  | 0.895105      |
| Pretrained\_aug    | 0.197190   | 0.946863       | 0.328382        | 0.891667            | 0.320064  | 0.895455      |


*(Note: The exact values might vary slightly based on training randomness.)*

## Contributing

Feel free to fork this repository, open issues, and submit pull requests. Any contributions are welcome! 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/ 
