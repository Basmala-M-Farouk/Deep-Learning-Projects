# 🧠 NeuroSegNet: Brain Tumor MRI Segmentation with Deep Learning

NeuroSegNet is a deep learning project focused on medical image segmentation. It is designed to identify and segment brain tumors from grayscale MRI scans. By using convolutional neural networks and pixel-wise mask prediction, the model can detect abnormal regions in brain scans with high precision.

This project is built from scratch using NumPy, OpenCV, and TensorFlow/Keras (U-Net or similar architecture assumed). It can serve as a foundation for clinical AI solutions, research applications, or educational demonstrations in the field of medical imaging.

---

## 📌 Project Objectives

- Segment tumor regions from MRI scans
- Build an end-to-end deep learning pipeline using raw image and mask data
- Enable reproducible training, visualization, and evaluation
- Lay the groundwork for further medical segmentation models

---

## 🗂 Dataset

**Source**: [Kaggle - Brain Tumor Segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

- 🧠 Grayscale MRI Images (128x128)
- 🎯 Ground Truth Masks (binary segmentation masks)
- Format: PNG/JPG images in `images/` and `masks/` folders

---

## ⚙️ Environment Setup

1. **Install dependencies:**
   ```bash
   pip install numpy opencv-python matplotlib scikit-learn tensorflow kagglehub
