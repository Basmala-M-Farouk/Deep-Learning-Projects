# 🏙️ CityScopeYOLO: Urban Scene Segmentation Using YOLOv8

CityScopeYOLO is a high-performance deep learning project that applies **YOLOv8 segmentation** models to complex urban scenes. Leveraging the **Cityscapes dataset** and the **Ultralytics YOLOv8 framework**, the project is designed for real-time object detection and segmentation in street-level imagery.

It integrates with **Roboflow** for streamlined dataset management and training pipeline setup, making it ideal for researchers, developers, and computer vision practitioners working on autonomous driving, smart city infrastructure, and traffic analysis.

---

## 🔍 Key Features

- 🚗 Instance and semantic segmentation of urban environments
- ⚡ Powered by Ultralytics YOLOv8 (fast & accurate)
- 📦 Dataset managed via Roboflow API
- 🧠 Real-time prediction-ready architecture
- 📉 Includes training, evaluation, and visualization tools

---

## 📂 Dataset: Cityscapes

- **Source**: [Cityscapes via Roboflow](https://public.roboflow.com/)
- **Classes**: Roads, pedestrians, vehicles, traffic lights, signs, etc.
- **Format**: YOLOv8-compatible annotation (`data.yaml`, `images/`, `labels/`)
- **Augmentations**: Provided by Roboflow preprocessing pipeline


---

## ⚙️ Installation & Setup

### 1. Clone the Repository
### 2. Install Dependencies
### 3. Authenticate & Download Dataset via Roboflow

```bash
git clone https://github.com/your-username/CityScopeYOLO.git
cd CityScopeYOLO

pip install roboflow ultralytics opencv-python matplotlib

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("cityscapes")
dataset = project.version("1").download("yolov8")

