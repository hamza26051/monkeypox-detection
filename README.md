# 🐵 Monkeypox Skin Lesion Classification Model

## 🌟 Project Overview
This project implements a custom **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** for the classification of skin lesions, aimed at distinguishing between different classes of images (presumably including "monkeypox" and others, based on the dataset structure).

The model features a custom architecture, utilizes data preprocessing and augmentation techniques, and employs a comprehensive training and evaluation workflow — demonstrating end-to-end deep learning model development.

---

## ✨ Key Features

- **Custom CNN Architecture:** Multiple Conv2D and MaxPooling2D layers for robust feature extraction.
- **Data Augmentation:** RandomFlip and RandomRotation layers to increase dataset variability and reduce overfitting.
- **Optimized Data Pipeline:** Uses `tf.data` utilities like `cache()`, `shuffle()`, and `prefetch(tf.data.AUTOTUNE)` for efficient input processing.
- **Custom Dataset Split:** Function `getdatasets` deterministically splits data into:
  - Training (80%)
  - Validation (10%)
  - Testing (10%)
- **Model Persistency:** Trained model is saved in Keras format (`.keras`) for deployment.

---

## 🛠️ Technical Specifications

**Libraries & Frameworks:**
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib

**Model Architecture Summary:**

| Layer (Type) | Output Shape | Parameters |
|---------------|---------------|-------------|
| Resizing (Sequential) | (224, 224, 3) | 0 |
| Augmentation (Sequential) | (224, 224, 3) | 0 |
| Convolutional Blocks (x6) | Reduces spatial dimensions | Varies |
| Flatten | (n,) | 0 |
| Dense Layers (x3) | Classification Head | Varies |
| Output Layer (Dense) | (4, softmax) | Varies |

> Multiple convolutional layers use (3,3) kernels with ReLU activation, followed by (2,2) MaxPooling for downsampling.

---

## ⚙️ Training Parameters

- **Epochs:** 180  
- **Optimizer:** Adam  
- **Loss Function:** sparse_categorical_crossentropy  
- **Batch Size:** 32 (default from `image_dataset_from_directory`)  
- **Input Image Size:** (224, 224)

---

## 🏃 Getting Started

### 1. Prerequisites
Install required dependencies:

```bash
pip install tensorflow numpy pandas matplotlib
```

### 2. Dataset Setup
Ensure your dataset is structured as follows:

```
/monkeyimages
    /Class_A (e.g., Monkeypox)
        img_a1.jpg
        img_a2.jpg
    /Class_B (e.g., Other_Rash)
        img_b1.jpg
        img_b2.jpg
    ...
```
> The model expects **4 classes** based on its final dense layer.

### 3. Run the Training Script

Run the model training and evaluation:

```bash
python <your_script_name>.py
```

---

## 🔍 Results and Evaluation

After training, the script outputs the overall test score:

```python
score = model.evaluate(testds)
```

It also visualizes predictions with:
- **Actual Class**
- **Predicted Class**
- **Confidence (% probability)**

---

## 🔮 Future Improvements

- Implement **Transfer Learning** using pre-trained models like **ResNet50** or **EfficientNetB0**.
- Integrate **Learning Rate Scheduler** or **Early Stopping** callbacks.
- Compute additional metrics such as **Precision**, **Recall**, and **F1-score** for each class.

---

📁 **Author:** Hamza Sheikh  
🎓 **Domain:** Deep Learning / Computer Vision  
💡 **Framework:** TensorFlow / Keras  
📦 **File Format:** `.keras` for model persistence  
