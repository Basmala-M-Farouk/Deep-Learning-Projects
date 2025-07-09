# 🧠🔬 NeuroSegNet: Brain Tumor Segmentation with Deep Learning 
## Overview

This repository features **NeuroSegNet**, a deep learning project dedicated to **segmenting brain tumors from MRI images**. Leveraging the `Brain Tumor Segmentation` dataset from Kaggle, this project establishes a robust pipeline for data acquisition, preprocessing, and visualization, laying the groundwork for advanced medical image analysis. Our aim is to develop a precise and efficient solution for automated tumor delineation, aiding in diagnosis and treatment planning. 💡

## Features

* **Brain Tumor Segmentation Dataset:** Automatically downloads and integrates the [Brain Tumor Segmentation dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation) from Kaggle using `kagglehub`. This dataset contains brain MRI images and their corresponding tumor masks. 📂
* **Automated Data Download:** Utilizes `kagglehub` to seamlessly download the dataset, eliminating manual download steps. ⬇️
* **Comprehensive Data Preprocessing:**
    * Loads brain MRI images and their associated tumor masks.
    * Resizes all images and masks to a uniform 128x128 pixels.
    * Normalizes pixel values to a [0, 1] range.
    * Reshapes data to include a channel dimension, suitable for grayscale images (single channel). ✂️
* **Data Organization:** Sets up clear directory paths for images (`images/`) and masks (`masks/`) within the downloaded dataset. 🗂️
* **Visual Verification:** Includes a utility function to display a side-by-side comparison of an MRI image and its corresponding tumor mask, ensuring correct data loading and preprocessing. 👀
* **Essential Libraries:** Developed using Python with key libraries such as `os`, `numpy`, `OpenCV (cv2)`, `matplotlib.pyplot`, `scikit-learn`, and `kagglehub`. 🐍

## Project Structure

* `NeuroSegNet Project.ipynb`: The primary Jupyter Notebook containing the full code for dataset download, data loading, preprocessing, and visualization. 📝

## Getting Started

### Prerequisites

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Matplotlib
* scikit-learn
* `kagglehub` library

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/NeuroSegNet.git](https://github.com/YourUsername/NeuroSegNet.git)
    cd NeuroSegNet
    ```
2.  Install the necessary Python libraries:
    ```bash
    pip install opencv-python numpy matplotlib scikit-learn kagglehub
    ```

### Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "NeuroSegNet Project.ipynb"
    ```
2.  Run through the cells to:
    * Automatically download and set up the dataset.
    * Load and preprocess the MRI images and masks.
    * Verify data integrity by visualizing sample images and masks.
    *(Note: This notebook focuses on data preparation. Subsequent steps for model building (e.g., U-Net), training, and evaluation would be added for a complete segmentation solution.)* ▶️

## Data Overview

The preprocessing steps result in `3064` images and `3064` masks, each resized to 128x128 pixels with a single channel. This ensures a consistent input format for deep learning models. 📈

## Contributing

We welcome contributions to expand NeuroSegNet! If you have ideas for model improvements, new features, or bug fixes, please fork the repository, open an issue, or submit a pull request. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/ 
