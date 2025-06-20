# 🧠 Brain Tumor Classification with Deep Learning

A Convolutional Neural Network (CNN) powered model to classify MRI brain scans into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

Built and trained in Google Colab using a Tesla T4 GPU on a public brain MRI dataset, achieving **98.6% test accuracy**.

---

## 🚀 Project Highlights

- 📊 **4-Class Classification** using MRI images
- 🔍 **Data Augmentation** and **Transfer Learning** with Xception
- 🧪 **Evaluation Metrics**: Accuracy, Precision, Recall
- 📉 Real-time plotting of **Loss**, **Accuracy**, **Precision**, and **Recall**
- 💾 Model saved in HDF5 (`.h5`) and Keras (`.keras`) formats
- 🖼️ Visual image prediction on user-selected MRI scans

---

## 🗂 Dataset Structure

Dataset from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
archive/

├── Training/

│ ├── glioma/

│ ├── meningioma/

│ ├── notumor/

│ └── pituitary/

└── Testing/

├── glioma/

├── meningioma/

├── notumor/

└── pituitary/

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Xception Model (Transfer Learning)
- Google Colab with Tesla T4 GPU
- Scikit-learn
- Seaborn & Matplotlib

---

## 📈 Model Performance

| Dataset      | Accuracy | Precision | Recall  | Loss   |
|--------------|----------|-----------|---------|--------|
| Train        | 100.00%  | 100.00%   | 100.00% | 0.0005 |
| Validation   | 97.55%   | 97.64%    | 97.55%  | 0.1110 |
| Test         | 98.63%   | 98.70%    | 98.63%  | 0.0471 |


---

## 📦 Installation & Usage

1. Clone the repo:
```bash
git clone https://github.com/Brum99/MRI_brain_tumor_prediction.git
cd MRI_brain_tumor_prediction

🧠 Author
Samuel Peterson
Machine Learning Engineer | LLM Evaluator | Data Scientist
🌐 brum99.github.io

📜 License
This project is licensed under the MIT License. See LICENSE for more information.
