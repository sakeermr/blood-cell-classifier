# 🩸 Blood Cell Classifier — Deep Learning Capstone Project

> **Gold Badge | BSc (Hons) Medical Laboratory Science**  
> Automated White Blood Cell (WBC) classification using Convolutional Neural Networks

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://blood-cell-classifier-kiqgclritqwub3tb4mrjs3.streamlit.app/)
[![GitHub](https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/sakeermr/blood-cell-classifier)

---

## 📌 Project Overview

This project builds an end-to-end deep learning application that classifies white blood cells (WBCs) from microscope images into four clinical categories. Automating WBC differential counting reduces manual lab work, speeds up diagnosis, and supports early detection of conditions like infections, allergies, and leukaemia.

**Problem:** Manual WBC differential count is time-consuming and prone to human error.  
**Solution:** A CNN-based image classifier that identifies WBC types from blood smear images in real time.

---

## 🎯 Classes

| Cell Type | Normal Range | Clinical Significance |
|-----------|-------------|----------------------|
| **NEUTROPHIL** | 50–70% | First responder to bacterial infections |
| **LYMPHOCYTE** | 20–40% | Key in viral immunity and antibody production |
| **MONOCYTE** | 2–8% | Differentiates into macrophages; fights chronic infections |
| **EOSINOPHIL** | 1–4% | Elevated in allergies and parasitic infections |

---

## 🏗️ Project Structure

```
blood-cell-classifier/
│
├── BloodCell_CNN_FIXED.ipynb        # Full training notebook (Google Colab)
├── app.py                           # Streamlit web application
│
├── blood_cell_classifier.h5         # Saved trained model (Custom CNN)
├── class_indices.json               # Class label mappings
├── model_metrics.json               # Model performance metrics
│
├── requirements.txt                 # Python dependencies
│
└── assets/
    ├── class_distribution.png       # EDA — class balance chart
    ├── sample_images.png            # Sample images per class
    ├── Custom_CNN_training_history.png
    ├── Custom_CNN_confusion_matrix_FINAL.png
    └── model_comparison_FINAL.png
```

---

## 🧠 Models Built & Compared

Two models were trained and evaluated on the same dataset:

### Model 1 — Custom CNN (Built from Scratch)
- **Architecture:** 3 convolutional blocks (Conv2D → BatchNorm → MaxPool → Dropout) + Dense head
- **Input size:** 224 × 224 × 3
- **Test Accuracy: 86.8%** ✅ *(Selected as Best Model)*

### Model 2 — MobileNetV2 (Transfer Learning)
- **Base:** MobileNetV2 pre-trained on ImageNet (frozen base → fine-tuned top 30 layers)
- **Phases:** Feature extraction → Fine-tuning → Extended fine-tuning
- **Test Accuracy: 64.8%**

| Metric | Custom CNN | MobileNetV2 |
|--------|-----------|-------------|
| Test Accuracy | **86.8%** ✅ | 64.8% |
| Val Accuracy | 99.7% | 96.8% |
| Strategy | From Scratch | Transfer Learning |
| Selected | **✅ Best Model** | ❌ |

> The Custom CNN outperformed MobileNetV2 on this dataset. Domain-specific microscopy images differ significantly from ImageNet's natural images, which limited the transfer learning benefit for MobileNetV2.

---

## 📊 Dataset

- **Source:** [Blood Cell Images — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) (`paultimothymooney/blood-cells`)
- **Total samples:** ~10,000 training images
- **Class balance:** Near-perfectly balanced (~2,500 images per class)
- **Image size:** 224 × 224 pixels (RGB)
- **Split:** Training / Validation (80/20 split) / Test set

---

## ⚙️ ML Pipeline

1. **Data Collection** — Kaggle blood cell dataset (real clinical microscopy images)
2. **EDA** — Class distribution charts, sample image visualization per class
3. **Preprocessing** — Pixel normalization (rescale 1/255), train-test split
4. **Augmentation** — Rotation, width/height shift, zoom, horizontal flip (training only)
5. **Model Training** — Custom CNN & MobileNetV2 with EarlyStopping, ReduceLROnPlateau
6. **Evaluation** — Accuracy, confusion matrix, classification report (Precision/Recall/F1)
7. **Model Saving** — `.h5` format with class indices JSON
8. **Deployment** — Streamlit web application

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sakeermr/blood-cell-classifier.git
cd blood-cell-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
tensorflow>=2.10.0
streamlit>=1.28.0
numpy>=1.23.0
Pillow>=9.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
```

> Full list in `requirements.txt`

---

## 🌐 Live Demo

**Streamlit Cloud:** [https://blood-cell-classifier-kiqgclritqwub3tb4mrjs3.streamlit.app/](https://blood-cell-classifier-kiqgclritqwub3tb4mrjs3.streamlit.app/)

Upload a blood smear microscopy image and the model will predict the WBC type along with the confidence score for each class.

---

## 📈 Results

**Best Model (Custom CNN) — Confusion Matrix Summary:**

| Actual \ Predicted | EOSINOPHIL | LYMPHOCYTE | MONOCYTE | NEUTROPHIL |
|--------------------|-----------|-----------|---------|-----------|
| **EOSINOPHIL** | 482 | 6 | 0 | 135 |
| **LYMPHOCYTE** | 0 | 620 | 0 | 0 |
| **MONOCYTE** | 0 | 0 | 466 | 154 |
| **NEUTROPHIL** | 33 | 0 | 0 | 591 |

**Overall Test Accuracy: 86.8%**

- LYMPHOCYTE classification is near-perfect (100% recall)
- Main confusion occurs between EOSINOPHIL/NEUTROPHIL and MONOCYTE/NEUTROPHIL — morphologically similar cell pairs

---

## 🔬 Training Details

| Parameter | Value |
|-----------|-------|
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Max epochs | 30 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| Platform | Google Colab (T4 GPU) |

---

## 📓 Notebook

Training was done in Google Colab with T4 GPU. Open `BloodCell_CNN_FIXED.ipynb` to reproduce all results.

Steps covered:
1. Library installation and imports
2. Kaggle dataset download and extraction
3. EDA and class distribution visualization
4. Data preprocessing and augmentation
5. Custom CNN architecture and training
6. MobileNetV2 transfer learning (3-phase training)
7. Model comparison and evaluation
8. Model export and asset saving

---

## 🏆 Badge Track

**🥇 GOLD BADGE** — Deep Learning (CNN Image Classification)

Difficulty: ⭐⭐⭐⭐⭐ Expert

---

## 📬 Contact

**GitHub:** [sakeermr](https://github.com/sakeermr/blood-cell-classifier)
