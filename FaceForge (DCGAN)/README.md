# ✨👨‍🎤 FaceForge: DCGAN Face Synthesis 

## Overview

This repository features **CelebGen**, a deep learning project dedicated to **Deep Convolutional Generative Adversarial Networks (DCGANs)** for high-quality **face synthesis**. We implement and train a DCGAN to generate new, realistic celebrity faces using the vast **CelebA dataset**. This project provides a hands-on demonstration of building, training, and evaluating both Generator and Discriminator networks, showcasing the fascinating capabilities of adversarial learning in creating lifelike imagery. 🖼️🚀

## Features

* **CelebA Dataset Integration:** Utilizes the large-scale [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (CelebFaces Attributes Dataset), containing over 200,000 celebrity images, which is expected to be loaded or extracted from Google Drive. 🧑‍🎤📸
* **Deep Convolutional GAN (DCGAN) Architecture:** Implements a robust DCGAN structure for stable and effective image generation:
    * **Generator:** Uses `Conv2DTranspose` for effective upsampling, `BatchNormalization`, `LeakyReLU`, and a `Tanh` activation for outputting images. 🎨
    * **Discriminator:** Employs `Conv2D` layers, `Dropout` for regularization, and `LeakyReLU` activations for robust real/fake classification. 🕵️‍♀️
* **Adversarial Training Loop:** Defines custom `generator_loss` and `discriminator_loss` functions and implements a `train_step` to alternate between optimizing the discriminator and the generator, driving the adversarial process. 🔄
* **Efficient Data Pipeline:** Preprocesses images by resizing them (e.g., to 64x64 pixels as commonly done in GANs) and normalizing pixel values to the range of [-1, 1], then uses `tf.data.Dataset` for optimized batching, shuffling, and prefetching. 🚀
* **Progress Visualization:** Generates and saves sample images at regular intervals during training, allowing for a visual assessment of the GAN's learning progress and the quality of generated faces. 👀
* **Model Persistence:** Includes functionality to save checkpoints of the Generator and Discriminator models, enabling continuation of training or deployment of trained models. 💾
* **TensorFlow 2.x:** Built entirely with TensorFlow 2.x for modern and efficient deep learning development. 🐍

## Project Structure

* `gans (1).ipynb`: The main Jupyter Notebook containing all the code for dataset setup, DCGAN model definitions, training loop implementation, and face generation. 📝

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow 2.x
* Matplotlib
* NumPy

### Installation

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/Basmala-M-Farouk/Deep-Learning-Projects.git](https://github.com/Basmala-M-Farouk/Deep-Learning-Projects.git) # Replace with your actual repo URL if different
    ```
2.  **Navigate into the GANs project directory:**
    ```bash
    cd Deep-Learning-Projects/FaceForge\ \(DCGAN\)
    # Adjust path if 'gans (1).ipynb' is in a differently named folder or has spaces (e.g., gans\ \(1\))
    ```
    * **Note:** If your folder name includes spaces (like `gans (1)`), you might need to escape them: `cd Deep-Learning-Projects/gans\ \(1\)`

3.  **Install the required libraries:**
    ```bash
    pip install tensorflow matplotlib numpy
    ```

### Usage

1.  **Prepare your dataset:** The notebook expects the CelebA dataset (e.g., as a `.zip` file or extracted images) to be accessible from your Google Drive, typically at a path like `/content/drive/MyDrive/CelebA/`. You will need to mount your Google Drive within the notebook.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "gans (1).ipynb"
    ```
3.  Run all cells in the notebook. This will:
    * Mount Google Drive and handle dataset extraction/loading.
    * Load and preprocess the CelebA images.
    * Define and compile the Generator and Discriminator networks.
    * Execute the DCGAN training loop, generating sample celebrity faces periodically. ▶️

## Results

During the training process, the DCGAN will generate increasingly realistic celebrity faces. The notebook provides visualizations of these generated images, showcasing the model's ability to learn and reproduce intricate facial features from the CelebA dataset. The loss values for both the generator and discriminator will also illustrate the adversarial training progress. 📈

*(Please refer to the `gans (1).ipynb` notebook for dynamic visualizations of generated faces and training progress.)*

## Contributing

Contributions are highly appreciated! If you have ideas for improving DCGAN architecture, training stability, or generating higher-quality faces, please feel free to fork this repository, open issues, or submit pull requests. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/
