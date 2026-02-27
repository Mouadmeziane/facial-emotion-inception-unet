# 😶 Facial Emotion Recognition — Inception ResNet V2 & U-Net

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-66.67%25-green)
![Classes](https://img.shields.io/badge/Emotion%20Classes-6-purple)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle)

---

## 📌 Overview

This project tackles **facial emotion recognition** on the **Autistic Children Emotions** dataset using two deep learning approaches:

- **Approach 1**: Hybrid model combining **Inception ResNet V2** + **U-Net**
- **Approach 2**: Standalone **Inception ResNet V2** with fine-tuning

The goal is to compare both architectures and understand the trade-offs between a segmentation-assisted approach and a pure classification approach for recognizing emotions in autistic children's facial expressions.

---

## 📊 Results

| Metric | Inception ResNet V2 + U-Net | Inception ResNet V2 Only |
|---|---|---|
| Test Accuracy | — | **66.67%** |
| Joy F1-score | — | 0.84 |
| Macro Avg F1 | — | 0.27 |

> ⚠️ Low scores on Anger and Fear are due to severe **class imbalance** (only 3 test samples each), not purely a model limitation.

---

## 🗂️ Dataset — Autistic Children Emotions

Created by **Dr. Fatma M. Talaat**, this dataset focuses on facial emotion recognition specifically in autistic children — an underexplored and important area in affective computing.

| Emotion | Label |
|---|---|
| Joy | 0 |
| Sadness | 1 |
| Fear | 2 |
| Anger | 3 |
| Surprise | 4 |
| Natural | 5 |

🔗 [Dataset on Kaggle](https://www.kaggle.com/datasets/your-link-here)

---

## 🏗️ Model Architectures

### Approach 1 — Inception ResNet V2 + U-Net
- **Inception ResNet V2** acts as the encoder, extracting rich multi-scale features
- **U-Net** decoder adds spatial precision via skip connections
- Best suited for tasks where localization of facial regions matters

### Approach 2 — Inception ResNet V2 Only
- Pre-trained on ImageNet (frozen base)
- Custom classification head added on top
- Input size: **299 × 299 px**
- Output: 6-class softmax

---

## ⚙️ Training Details

- Optimizer: Adam / SGD (tuned)
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping
- Data Augmentation: Applied via `ImageDataGenerator`
- Platform: Kaggle (GPU accelerated)

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/Mouadmeziane/facial-emotion-inception-unet.git
cd facial-emotion-inception-unet
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset** from Kaggle and place it under:
```
data/
├── Train/
│   ├── joy/
│   ├── sadness/
│   ├── fear/
│   ├── anger/
│   ├── surprise/
│   └── Natural/
└── Test/
    └── ...
```

4. **Open the notebook**
```bash
jupyter notebook Inception_ResNet_V2_and_U-Net_for_Facial_Emotions_Classification.ipynb
```

---

## 🛠️ Tech Stack

- Python 3.8
- TensorFlow / Keras
- Keras Tuner
- Scikit-learn
- OpenCV
- Seaborn / Matplotlib
- DEAP (Evolutionary Algorithms)
- NumPy

---

## 📈 What I'd Improve Next

- Collect or augment more samples for minority classes (Anger, Fear, Surprise)
- Apply class weights during training to handle imbalance
- Unfreeze top layers of Inception ResNet V2 for deeper fine-tuning
- Try EfficientNet or Vision Transformers for comparison
- Add Grad-CAM visualizations to interpret model predictions

---

## 📬 Contact

📧 mouadmeziane28@gmail.com
