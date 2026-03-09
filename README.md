# Brain-Tumor-Detection-from-MRI-using-CNN-LSTM
Project Motivation

Early detection of brain tumors is critical for effective treatment. MRI scans are widely used for diagnosis, but manual analysis by radiologists can be time-consuming and subjective.
The model combines Convolutional Neural Networks (CNN) for extracting spatial features and Long Short-Term Memory (LSTM) networks for learning deeper feature relationships. The goal was to evaluate whether a hybrid architecture could improve classification accuracy compared to traditional CNN models.

Project Overview
This project implements a cutting-edge **Hybrid CNN-LSTM architecture** for automated brain tumor detection from MRI images, achieving an impressive **98.47% accuracy**. The model combines:

- **Convolutional Neural Networks (CNN)** for spatial feature extraction from MRI scans
- **Long Short-Term Memory (LSTM)** networks for sequential pattern recognition
- **Advanced data augmentation** for robust generalization
- **Production-ready code** with comprehensive documentation

Dataset

Two publicly available datasets were used.
1. **Br35h Dataset** (3,000 images)
   - [Download from Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
   - High-quality MRI scans
   - Balanced classes

2. **Sartaj Dataset** (1,311 images)
   - [Download from Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
   - Multiple MRI modalities
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

### Training

```python
from brain_tumor_cnn_lstm import BrainTumorCNNLSTM, DataLoader

# Load data
loader = DataLoader(data_dir='path/to/dataset', img_size=150)
X, y = loader.load_dataset(dataset_name='br35h')

# Build model
model = BrainTumorCNNLSTM(img_size=150, num_classes=2)
model.build_model()

# Train
history = model.train(X_train, y_train, X_val, y_val, epochs=50)

# Evaluate
metrics = model.evaluate(X_test, y_test)
```

### Inference

**Single Image Prediction:**

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('outputs/brain_tumor_cnn_lstm_model.h5')

# Load and preprocess image
img = cv2.imread('mri_scan.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (150, 150))
img = img.astype('float32') / 255.0
img = img.reshape(1, 150, 150, 1)

# Predict
prediction = model.predict(img)
class_idx = np.argmax(prediction[0])
confidence = prediction[0][class_idx] * 100

classes = ['Brain Tumor', 'Healthy']
print(f"Prediction: {classes[class_idx]} ({confidence:.2f}%)")
```

**Batch Prediction:**

```bash
python inference.py \
    --model outputs/brain_tumor_cnn_lstm_model.h5 \
    --folder ./test_images/ \
    --output predictions.csv
```

## 📚 Documentation

Comprehensive documentation is available:

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - project overview |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Detailed step-by-step tutorial |
| [CHEAT_SHEET.md](CHEAT_SHEET.md) | Quick reference guide |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | How to run and deploy |
| [Brain_Tumor_Detection_Colab.ipynb](Brain_Tumor_Detection_Colab.ipynb) | Interactive Colab notebook |

Limitations
Although the model achieves high accuracy, several limitations remain:
Dataset size is relatively small
MRI scans come from different sources with varying quality
The model performs binary classification only
Further validation on clinical datasets would be required for real-world deployment.

## 🌟 Star History
If you find this project useful, please consider giving it a ⭐!

License
This project is released under the MIT License.
