# GestureNet: Deep Learning for Rock-Paper-Scissors Recognition

This project applies deep learning models—MLP and CNN—to classify images from the [Rock-Paper-Scissors dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors). It demonstrates an end-to-end pipeline including data loading, preprocessing, model training, and evaluation using TensorFlow and TensorFlow Datasets.

## 🧠 Models Used
- **MLP (Multilayer Perceptron)**: A simple dense network with dropout layers.
- **CNN (Convolutional Neural Network)**: Uses convolutional and pooling layers to extract spatial features.

## 📁 Dataset
- **Source**: TensorFlow Datasets
- **Classes**: Rock, Paper, Scissors
- **Preprocessing**: Images resized to 150x150, normalized to [0, 1] range.

## 🚀 How to Run

### Requirements
- Python ≥ 3.7
- TensorFlow ≥ 2.x
- Matplotlib
- Pandas

### Setup
```bash
pip install tensorflow matplotlib pandas tensorflow-datasets
