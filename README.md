# Cats-Dogs-Prediction-CNN
A deep learning project for Dogs vs Cats image classification using Transfer Learning with ResNet50. Includes data preprocessing with augmentation, fine-tuning, and regularization. Achieves 99.15% training accuracy and 98.72% validation accuracy. Built with TensorFlow/Keras.


# 🐶🐱 Dogs vs Cats Classification using ResNet50

This project is a **Deep Learning Image Classification model** that classifies images of **dogs and cats** using **Transfer Learning with ResNet50**.  
The model achieves **99.15% training accuracy** and **98.72% validation accuracy**.

---

## 📌 Dataset
The dataset used in this project is from [Kaggle Dogs vs Cats](https://www.kaggle.com/), structured as:
```
train/
    cats/
    dogs/
validation/
    cats/
    dogs/
```

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Applied `ImageDataGenerator` with augmentation (rotation, shift, zoom, flip, rescale).
   - Created training and validation generators.

2. **Model Architecture**
   - Base model: **ResNet50** (pre-trained on ImageNet, `include_top=False`).
   - Global Average Pooling Layer.
   - Dense layer (128 units, ReLU activation).
   - Dropout (0.3) for regularization.
   - Output layer: Dense(1, sigmoid).

3. **Training**
   - Optimizer: Adam (`lr=1e-4`).
   - Loss: Binary Crossentropy.
   - Metrics: Accuracy.
   - Callbacks: EarlyStopping, ReduceLROnPlateau.

---

## 📊 Results
- **Training Accuracy:** 99.15%
- **Validation Accuracy:** 98.72%

Confusion matrix and evaluation confirmed strong generalization with minimal overfitting.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/WaleedAfridi-1/Cats-Dogs-Prediction-CNN.git
   cd dogs-vs-cats-resnet50
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train.py
   ```

4. Evaluate the model / Predict:
   ```bash
   python predict.py --image path_to_image.jpg
   ```

---

## 📌 Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn

---

## 📷 Sample Predictions
(Add sample prediction images here)

---

## ✨ Author
- **Waleed Afridi**  

