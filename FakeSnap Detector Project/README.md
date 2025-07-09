# 📸🕵️‍♀️ FakeSnap Detector: Unmasking AI-Generated Faces 

## Overview

This repository houses the **FakeSnap Detector** project, a deep learning solution designed to distinguish between **real and AI-generated (synthetic) facial images**. Leveraging the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) from Kaggle, this project demonstrates a complete pipeline for image classification, including data acquisition, robust preprocessing with augmentation, and visual verification of processed data. Our goal is to contribute to the detection of deepfakes and digitally altered content. 🤖🚫

## Features

* **CIFAKE Dataset Integration:** Automatically downloads and extracts the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) directly from Kaggle, which contains a large collection of real and AI-generated facial images. 📂
* **Kaggle API Setup:** Includes instructions and code for setting up Kaggle API access to facilitate seamless dataset downloads. 🔑
* **Robust Data Preprocessing:** Prepares images for deep learning by resizing them to 224x224 pixels and rescaling pixel values to a [0, 1] range for optimal model performance. ⚙️
* **Advanced Data Augmentation:** Applies various augmentation techniques such as horizontal flips, random zooms, and rotations (up to 20 degrees) using `ImageDataGenerator` to enhance model generalization and prevent overfitting. 📊
* **Train/Validation Split:** Splits the dataset into 80% for training and 20% for validation to ensure robust model evaluation. 🧪
* **Image Visualization:** Provides code to visualize sample augmented images, allowing for verification of the data pipeline and confirming correct labeling before model training. 👀
* **TensorFlow & Keras:** Built with the powerful TensorFlow 2.x and Keras API for efficient deep learning model development and training. 🐍

## Project Structure

* `FakeSnap Detector Project.ipynb`: The main Jupyter Notebook containing all the code for dataset setup, data preprocessing, augmentation, and visualization. 📝

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow 2.x
* Kaggle API
* Matplotlib
* Numpy

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/FakeSnap-Detector.git](https://github.com/YourUsername/FakeSnap-Detector.git)
    cd FakeSnap-Detector
    ```
2.  Install the required libraries (ensure you have your `kaggle.json` file ready for Kaggle API setup):
    ```bash
    pip install tensorflow matplotlib numpy kaggle
    ```
3.  Upload your `kaggle.json` file when prompted in the notebook (or place it in `~/.kaggle/` and set permissions `chmod 600 ~/.kaggle/kaggle.json`).

### Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook FakeSnap Detector Project.ipynb
    ```
2.  Follow the instructions in the notebook cells to:
    * Set up Kaggle API access.
    * Download and extract the `CIFAKE` dataset.
    * Prepare data with augmentation.
    * Visualize sample images.
    * (Further steps for model building and training will follow in a complete deep learning project). ▶️

## Results

This project establishes the foundational steps for an image classification task, focusing on data preparation and augmentation. The successful setup allows for the generation of `80,000` training images and `20,000` validation images belonging to 2 classes (real/fake). The visualizations confirm that the data pipeline is correctly augmenting and preparing images for a deep learning model. 📈

*(Note: This notebook focuses on data preparation. Further steps for model definition, training, and evaluation would be added for a complete end-to-end solution.)*

## Contributing

Contributions are highly encouraged! Feel free to fork this repository, open issues for bug reports or feature suggestions, and submit pull requests. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/ 
