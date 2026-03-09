# Brain-Tumor-Detection-from-MRI-using-CNN-LSTM
Project Motivation

Early detection of brain tumors is critical for effective treatment. MRI scans are widely used for diagnosis, but manual analysis by radiologists can be time-consuming and subjective.

This project explores how deep learning can assist in automated tumor detection by learning patterns directly from MRI images.

The model combines Convolutional Neural Networks (CNN) for extracting spatial features and Long Short-Term Memory (LSTM) networks for learning deeper feature relationships. The goal was to evaluate whether a hybrid architecture could improve classification accuracy compared to traditional CNN models.

Project Overview

This repository contains a deep learning pipeline for binary classification of brain MRI scans:


Dataset

Two publicly available datasets were used.
Br35h Brain Tumor Dataset
~3000 MRI images
Balanced tumor and non-tumor classes
Sartaj Brain Tumor Dataset
1311 MRI images
Additional MRI scans for improved diversity
Combined dataset size: ~4300 images

Dataset structure:

dataset
│
├── yes
│   ├── image1.jpg
│   ├── image2.jpg
└── no
    ├── image1.jpg
    ├── image2.jpg
    
Data Preprocessing

Several preprocessing steps were applied before training:
Conversion to grayscale
Resizing images to 150 × 150
Pixel normalization to [0,1]
Data augmentation
Augmentation techniques used:
Rotation
Horizontal and vertical flipping
Width and height shifts
Zoom transformations

These steps helped improve model generalization and reduce overfitting.

## 📈 Performance

### Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.47% |
| **Precision** | 97.9% |
| **Recall** | 97.6% |
| **F1-Score** | 97.7% |
| **ROC-AUC** | 99.73% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Brain Tumor** | 90% | 98% | 94% |
| **Healthy** | 96% | 83% | 89% |

### Comparison with Other Methods

| Method | Accuracy | Reference |
|--------|----------|-----------|
| Hybrid Ensemble (KNN-RF-DT) | 97.30% | Literature |
| Ensemble (DT, KNN, SVM) | 97.91% | Literature |
| Ensemble (Inception-v3, ResNet101, DenseNet201) | 97.14% | Literature |
| **CNN-LSTM (This Work)** | **98.47%** | ✅ |

---

##  Architecture

### Model Structure

```
Input (150×150×1 grayscale MRI)
         ↓
┌─────────────────────────┐
│   CNN Block 1           │
│   • Conv2D (64, 5×5)    │
│   • MaxPooling (2×2)    │
│   • Dropout (0.25)      │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│   CNN Block 2           │
│   • Conv2D (128, 3×3)   │
│   • MaxPooling (2×2)    │
│   • Dropout (0.25)      │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│   CNN Block 3           │
│   • Conv2D (128, 3×3)   │
│   • MaxPooling (2×2)    │
│   • Dropout (0.30)      │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│   CNN Block 4           │
│   • Conv2D (256, 3×3)   │
│   • MaxPooling (2×2)    │
│   • Dropout (0.30)      │
└─────────────────────────┘
         ↓
    Reshape for LSTM
         ↓
┌─────────────────────────┐
│   LSTM Layer            │
│   • 256 units           │
│   • Dropout (0.2)       │
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│   Dense Layers          │
│   • Dense (1024)        │
│   • Dropout (0.2)       │
│   • Output (2)          │
└─────────────────────────┘
         ↓
Output (Tumor / Healthy)
```

**Total Parameters**: 3,405,954 (3.79M trainable)
Key details:
Multiple convolution layers extract spatial patterns from MRI scans.
Feature maps are reshaped into sequences for the LSTM.
The LSTM layer captures relationships between extracted features.
Final dense layers perform classification.

Running the Project
Clone the Repository
git clone https://github.com/yourusername/Brain-Tumor-Detection-CNN-LSTM.git
cd Brain-Tumor-Detection-CNN-LSTM
Create Virtual Environment
python -m venv venv

Activate environment
Linux / Mac
source venv/bin/activate
Windows
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Training the Model

Run the training script:

python brain_tumor_cnn_lstm.py
This will:
Load MRI images
Apply preprocessing and augmentation
Train the CNN-LSTM model
Save the trained model
Making Predictions
Example inference code:
from tensorflow.keras.models import load_model
import cv2
import numpy as np
model = load_model("outputs/model.h5")
img = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(150,150))
img = img/255.0
img = img.reshape(1,150,150,1)
prediction = model.predict(img)
classes = ["Tumor","Healthy"]
print(classes[np.argmax(prediction)])
Project Structure
Brain-Tumor-Detection-CNN-LSTM
dataset/
    yes/
    no/
brain_tumor_cnn_lstm.py
inference.py
requirements.txt
README.md
notebooks/
Limitations
Although the model achieves high accuracy, several limitations remain:
Dataset size is relatively small
MRI scans come from different sources with varying quality
The model performs binary classification only
Further validation on clinical datasets would be required for real-world deployment.

Future Work
Possible extensions of this project:
Multi-class tumor classification
Grad-CAM visualizations for model explainability
3D MRI volume analysis
Web application for MRI upload and prediction
Model deployment using Docker or cloud APIs
Skills Demonstrated

This project demonstrates experience with:
Deep Learning
Computer Vision
Medical Image Classification
CNN Architectures
LSTM Networks
Data Augmentation
Model Evaluation Metrics
TensorFlow / Keras
Python ML pipelines

License
This project is released under the MIT License.
