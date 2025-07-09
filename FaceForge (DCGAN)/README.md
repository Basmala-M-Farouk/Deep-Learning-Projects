#🐾✨ FaceForge (DCGAN): Generating Paw-some Images 

## Overview

This repository features **GANiverse**, a deep learning project dedicated to **Generative Adversarial Networks (GANs)** for image synthesis. We implement and train a Conditional GAN (CGAN) to generate new, realistic images, specifically focusing on generating images reminiscent of cats and dogs using the widely-known PetImages dataset. This project provides a hands-on demonstration of building, training, and evaluating both Generator and Discriminator networks, showcasing the fascinating capabilities of adversarial learning. 🖼️🚀

## Features

* **PetImages Dataset:** Utilizes the [Dogs vs. Cats dataset (PetImages)](https://www.kaggle.com/datasets/salader/dogs-vs-cats), a popular collection of animal images, accessed and extracted from Google Drive. 🐕🐈
* **Complete GAN Architecture:** Implements both the Generator and Discriminator networks from scratch:
    * **Generator:** Uses `Conv2DTranspose` for effective upsampling, `BatchNormalization`, `LeakyReLU`, and a `Tanh` activation for outputting images. 🎨
    * **Discriminator:** Employs `Conv2D` layers, `Dropout` for regularization, and `LeakyReLU` activations for robust real/fake classification. 🕵️‍♀️
* **Adversarial Training Loop:** Defines custom `generator_loss` and `discriminator_loss` functions and implements a `train_step` to alternate between optimizing the discriminator and the generator, driving the adversarial process. 🔄
* **Efficient Data Pipeline:** Preprocesses images by resizing them to 64x64 pixels and normalizing pixel values to the range of [-1, 1], then uses `tf.data.Dataset` for optimized batching, shuffling, and prefetching. 🚀
* **Progress Visualization:** Generates and saves sample images at regular intervals during training, allowing for a visual assessment of the GAN's learning progress and the quality of generated images. 👀
* **Model Persistence:** Includes functionality to save checkpoints of the Generator and Discriminator models, enabling continuation of training or deployment of trained models. 💾
* **TensorFlow 2.x:** Built entirely with TensorFlow 2.x for modern and efficient deep learning development. 🐍

## Project Structure

* `gans (1).ipynb`: The main Jupyter Notebook containing all the code for dataset setup, GAN model definitions, training loop implementation, and image generation. 📝

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
    cd Deep-Learning-Projects/FaceForge\ \(DCGAN\) # Adjust path if 'gans (1).ipynb' is in a differently named folder
    ```
    * **Note:** If your folder name includes spaces (like `gans (1)`), you might need to escape them: `cd Deep-Learning-Projects/gans\ \(1\)`

3.  **Install the required libraries:**
    ```bash
    pip install tensorflow matplotlib numpy
    ```

### Usage

1.  **Prepare your dataset:** Ensure you have the `PetImages.zip` file (or a folder containing `Cat` and `Dog` subdirectories) placed in your Google Drive, accessible via the specified path (e.g., `/content/drive/MyDrive/PetImages.zip`).
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "gans (1).ipynb"
    ```
3.  Run all cells in the notebook. This will:
    * Mount Google Drive and extract the dataset.
    * Load and preprocess the images.
    * Define and compile the Generator and Discriminator.
    * Execute the GAN training loop, generating sample images periodically. ▶️

## Results

During the training process, the GAN will generate increasingly realistic images. The notebook provides visualizations of these generated images, showcasing the model's ability to learn and reproduce patterns from the input dataset. The loss values for both the generator and discriminator will also illustrate the adversarial training progress. 📈

*(Please refer to the `gans (1).ipynb` notebook for dynamic visualizations of generated images and training progress.)*

## Contributing

Contributions are highly appreciated! If you have ideas for improving GAN architecture, training stability, or generating higher-quality images, please feel free to fork this repository, open issues, or submit pull requests. 🤝

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 📄

## Contact 📧

https://www.linkedin.com/in/basmala-mohamed-farouk-079588223/
