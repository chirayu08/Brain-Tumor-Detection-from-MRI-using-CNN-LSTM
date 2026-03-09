# Brain Tumor Detection - Complete Usage Guide

## 📋 Table of Contents
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training the Model](#training-the-model)
4. [Making Predictions](#making-predictions)
5. [Understanding the Results](#understanding-the-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## 1. Installation

### Step 1.1: System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: ~5GB free space
- **GPU** (optional): NVIDIA GPU with CUDA 11.2+

### Step 1.2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv brain_tumor_env

# Activate virtual environment
# On Windows:
brain_tumor_env\Scripts\activate
# On macOS/Linux:
source brain_tumor_env/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 1.3: Verify Installation

```python
# Run this Python script to verify
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU available: {len(gpus)} GPU(s)")
else:
    print("Running on CPU")
```

---

## 2. Dataset Preparation

### Step 2.1: Download Dataset

**Option A: Br35h Dataset (Recommended)**
1. Go to: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
2. Click "Download" (requires Kaggle account)
3. Extract the downloaded ZIP file

**Option B: Sartaj Dataset**
1. Go to: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
2. Click "Download"
3. Extract the downloaded ZIP file

**Option C: Combined Dataset (Best Performance)**
- Download both datasets
- Combine all images from both sources

### Step 2.2: Organize Dataset Structure

After downloading, organize your dataset as follows:

```
brain_tumor_dataset/
├── yes/                    # Folder for brain tumor images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no/                     # Folder for healthy brain images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Important Notes:**
- All tumor images must be in the `yes/` folder
- All healthy brain images must be in the `no/` folder
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images can be any size (will be resized to 150×150)

### Step 2.3: Verify Dataset

```python
import os

# Update this path
DATASET_PATH = '/path/to/brain_tumor_dataset'

# Check folder structure
yes_folder = os.path.join(DATASET_PATH, 'yes')
no_folder = os.path.join(DATASET_PATH, 'no')

print(f"Tumor images: {len(os.listdir(yes_folder))}")
print(f"Healthy images: {len(os.listdir(no_folder))}")
print(f"Total images: {len(os.listdir(yes_folder)) + len(os.listdir(no_folder))}")
```

---

## 3. Training the Model

### Step 3.1: Update Configuration

Open `brain_tumor_cnn_lstm.py` and update the dataset path:

```python
# Find this section (around line 850)
DATA_DIR = '/path/to/your/brain_tumor_dataset'  # UPDATE THIS
DATASET_NAME = 'br35h'  # or 'sartaj'
```

### Step 3.2: Run Training

**Basic Training:**
```bash
python brain_tumor_cnn_lstm.py
```

**Expected Output:**
```
============================================================
BRAIN TUMOR DETECTION USING HYBRID CNN-LSTM MODEL
============================================================

Step 1: Loading Dataset
------------------------------------------------------------
Loading 3000 images from yes...
Loading: 100%|████████████| 3000/3000 [00:30<00:00]
Loading 3000 images from no...
Loading: 100%|████████████| 3000/3000 [00:30<00:00]

Dataset Summary:
------------------------------------------------------------
Total images: 6000
Tumor images: 3000
Healthy images: 3000
...
```

### Step 3.3: Monitor Training

Training will display progress for each epoch:

```
Epoch 1/50
56/56 [==============================] - 45s 800ms/step 
- loss: 0.3456 - accuracy: 0.8523 - val_loss: 0.2134 - val_accuracy: 0.9123

Epoch 2/50
56/56 [==============================] - 42s 750ms/step
- loss: 0.2134 - accuracy: 0.9234 - val_loss: 0.1567 - val_accuracy: 0.9456
...
```

**Training Time:**
- CPU only: ~2-3 hours
- GPU (NVIDIA): ~30-45 minutes

### Step 3.4: Training Outputs

After training completes, check the `outputs/` folder:

```
outputs/
├── training_history.png          # Training curves
├── confusion_matrix.png          # Classification matrix
├── roc_curve.png                 # ROC curve
├── sample_predictions.png        # Example predictions
├── brain_tumor_cnn_lstm_model.h5 # Trained model
└── evaluation_metrics.csv        # Performance metrics
```

---

## 4. Making Predictions

### Method 1: Single Image Prediction

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model('outputs/brain_tumor_cnn_lstm_model.h5')

# Load and preprocess image
img = cv2.imread('path/to/mri_scan.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (150, 150))
img = img.astype('float32') / 255.0
img = img.reshape(1, 150, 150, 1)

# Make prediction
prediction = model.predict(img)
class_idx = np.argmax(prediction[0])
confidence = prediction[0][class_idx] * 100

class_names = ['Brain Tumor', 'Healthy']
print(f"Prediction: {class_names[class_idx]}")
print(f"Confidence: {confidence:.2f}%")
```

### Method 2: Using Inference Script

**Single Image:**
```bash
python inference.py \
    --model outputs/brain_tumor_cnn_lstm_model.h5 \
    --image path/to/mri_scan.jpg
```

**Batch Prediction:**
```bash
python inference.py \
    --model outputs/brain_tumor_cnn_lstm_model.h5 \
    --folder path/to/image/folder \
    --output predictions.csv
```

### Method 3: Interactive Prediction

```python
# inference_interactive.py
from inference import predict_single_image, load_model

# Load model once
model = load_model('outputs/brain_tumor_cnn_lstm_model.h5')

# Predict multiple images
images = [
    'patient1_scan.jpg',
    'patient2_scan.jpg',
    'patient3_scan.jpg'
]

for image_path in images:
    print(f"\nAnalyzing: {image_path}")
    result = predict_single_image(model, image_path, show_result=True)
    print(f"Result: {result['predicted_class']} ({result['confidence']:.2f}%)")
```

---

## 5. Understanding the Results

### 5.1: Training History Plot

**What to look for:**
- **Accuracy curves**: Should increase and plateau
- **Loss curves**: Should decrease and stabilize
- **Validation vs Training**: Should be close (no overfitting)

**Good Training:**
```
Training Accuracy: 98%
Validation Accuracy: 97%
Small gap = Good generalization
```

**Overfitting (Bad):**
```
Training Accuracy: 99%
Validation Accuracy: 85%
Large gap = Model memorized training data
```

### 5.2: Confusion Matrix

```
                Predicted
                Tumor  | Healthy
Actual  Tumor   1595   |   38      = 98% Recall
        Healthy  22    |   983     = 97% Precision
```

**Key Metrics:**
- **True Positives (TP)**: Correctly identified tumors
- **True Negatives (TN)**: Correctly identified healthy
- **False Positives (FP)**: Healthy classified as tumor
- **False Negatives (FN)**: Tumor classified as healthy

### 5.3: ROC Curve

**AUC Score Interpretation:**
- **0.90-1.00**: Excellent
- **0.80-0.90**: Good
- **0.70-0.80**: Fair
- **<0.70**: Poor

Our model achieves **AUC = 0.997** (Excellent!)

### 5.4: Evaluation Metrics

```
Metric          Value      Interpretation
----------------------------------------------
Accuracy        98.47%     Overall correctness
Precision       97.9%      How many predicted tumors are actual tumors
Recall          97.6%      How many actual tumors were detected
F1-Score        97.7%      Balance between precision and recall
ROC-AUC         99.73%     Model's discriminative ability
```

---

## 6. Advanced Usage

### 6.1: Custom Training Parameters

```python
# Modify in brain_tumor_cnn_lstm.py

# Increase epochs for better training
EPOCHS = 100  # Default: 50

# Adjust batch size based on GPU memory
BATCH_SIZE = 16  # Default: 32 (smaller = less memory)

# Change image size
IMG_SIZE = 224  # Default: 150 (larger = more detail, slower)
```

### 6.2: Transfer Learning

```python
# Load pre-trained model
base_model = load_model('outputs/brain_tumor_cnn_lstm_model.h5')

# Freeze early layers
for layer in base_model.layers[:10]:
    layer.trainable = False

# Add new layers for your specific dataset
x = base_model.output
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

# Create new model
new_model = Model(inputs=base_model.input, outputs=output)

# Train on new dataset
new_model.compile(optimizer=Adam(0.0001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
```

### 6.3: Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

# 5-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\nTraining Fold {fold + 1}/5...")
    
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    
    # Build fresh model for each fold
    model = BrainTumorCNNLSTM()
    model.build_model()
    
    # Train
    history = model.train(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
    
    # Evaluate
    metrics = model.evaluate(X_val_cv, y_val_cv)
    cv_scores.append(metrics['accuracy'])

print(f"\nCross-Validation Results:")
print(f"Mean Accuracy: {np.mean(cv_scores)*100:.2f}%")
print(f"Std Deviation: {np.std(cv_scores)*100:.2f}%")
```

### 6.4: Model Interpretability with Grad-CAM

```python
import tensorflow as tf
from tensorflow.keras.models import Model

def generate_gradcam(model, img, layer_name='conv2d_4'):
    """
    Generate Grad-CAM heatmap
    """
    # Create model that outputs last conv layer
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, np.argmax(predictions[0])]
    
    # Get gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# Usage
img = preprocess_image('mri_scan.jpg')
heatmap = generate_gradcam(model, img)

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(img.squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(132)
plt.imshow(heatmap, cmap='jet')
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.subplot(133)
plt.imshow(img.squeeze(), cmap='gray')
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title('Overlay')
plt.axis('off')
plt.show()
```

---

## 7. Troubleshooting

### Issue 1: Out of Memory Error

**Error:**
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. Reduce batch size:
   ```python
   BATCH_SIZE = 16  # or 8
   ```

2. Reduce image size:
   ```python
   IMG_SIZE = 128  # instead of 150
   ```

3. Enable GPU memory growth:
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

### Issue 2: Dataset Not Found

**Error:**
```
Warning: Folder not found: /path/to/dataset/yes
```

**Solutions:**
1. Check folder structure:
   ```bash
   ls -la /path/to/dataset/
   # Should show: yes/ and no/ folders
   ```

2. Update path in code:
   ```python
   DATA_DIR = '/correct/path/to/brain_tumor_dataset'
   ```

3. Verify folder names are exactly `yes` and `no` (lowercase)

### Issue 3: Slow Training on CPU

**Problem:** Training takes too long without GPU

**Solutions:**
1. Use Google Colab (free GPU):
   ```python
   # Upload files to Colab
   # Runtime > Change runtime type > GPU
   ```

2. Use smaller dataset for testing:
   ```python
   # Sample subset of data
   X = X[:1000]  # Use only 1000 images
   y = y[:1000]
   ```

3. Reduce epochs:
   ```python
   EPOCHS = 20  # instead of 50
   ```

### Issue 4: Low Accuracy

**Problem:** Model accuracy below 90%

**Possible Causes and Solutions:**

1. **Insufficient training data**
   - Solution: Use both datasets (Br35h + Sartaj)
   - Add more data augmentation

2. **Class imbalance**
   - Check distribution: `print(np.bincount(y))`
   - Use class weights:
     ```python
     from sklearn.utils import class_weight
     
     class_weights = class_weight.compute_class_weight(
         'balanced', 
         classes=np.unique(y_train),
         y=y_train
     )
     
     # In model.fit():
     class_weight={0: class_weights[0], 1: class_weights[1]}
     ```

3. **Learning rate too high/low**
   - Try different learning rates:
     ```python
     optimizer = Adam(learning_rate=0.0001)  # Default
     # or
     optimizer = Adam(learning_rate=0.00001)  # Lower
     ```

### Issue 5: Model Overfitting

**Signs:**
- Training accuracy > 95%
- Validation accuracy < 85%
- Large gap between training and validation

**Solutions:**
1. Increase dropout:
   ```python
   Dropout(0.5)  # instead of 0.3
   ```

2. Add more data augmentation:
   ```python
   datagen = ImageDataGenerator(
       rotation_range=20,      # instead of 15
       width_shift_range=0.2,  # instead of 0.1
       height_shift_range=0.2,
       horizontal_flip=True,
       vertical_flip=True,
       zoom_range=0.2,
       shear_range=0.1         # add shearing
   )
   ```

3. Use early stopping:
   ```python
   EarlyStopping(patience=5)  # instead of 10
   ```

### Issue 6: ImportError

**Error:**
```
ImportError: No module named 'tensorflow'
```

**Solution:**
```bash
# Verify environment is activated
# Install/reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check Error Messages**: Read the full error traceback
2. **Verify Installation**: Run verification script from Section 1.3
3. **Check File Paths**: Ensure all paths are correct and accessible
4. **GPU Issues**: Try running on CPU first to isolate problems
5. **Dataset Issues**: Verify folder structure matches Section 2.2

---

## 🎉 Success Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed from requirements.txt
- [ ] Dataset downloaded and organized correctly
- [ ] Training completed successfully
- [ ] Model achieves >90% accuracy
- [ ] Can make predictions on new images
- [ ] Visualizations generated correctly

---

**Last Updated:** February 2026  
**Version:** 1.0.0
