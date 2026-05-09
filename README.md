# 🌿 PredictiveProject1  
## Plant Disease Classification Using Machine Learning  

### 👥 Team Members  
- Aparna Sreenivasan  
- Ayana Santhosh Khan 
- Shiha Shajahan  

### 📘 Course  
Predictive Analytics  

---

# 🌱 Problem Statement & Motivation  
Plant diseases significantly reduce crop quality and agricultural productivity. Farmers often struggle to identify diseases at an early stage, leading to major yield loss.

This project aims to build a machine learning-based system that classifies plant leaf images as healthy or diseased, enabling early detection and better agricultural decision-making.

---

# 📊 Dataset Description  
We used the PlantVillage dataset, a widely used dataset for plant disease classification.

### 📌 Dataset Details:
- Source: PlantVillage Dataset  
- Size: ~54,000 images  
- Classes: 38 plant disease categories  
- Type: Labeled leaf images  

### 📌 Example Classes:
- Healthy  
- Bacterial Spot  
- Early Blight  
- Late Blight  
- Leaf Mold  
- Mosaic Virus  

---

# ⚙️ Methodology (Machine Learning Pipeline)

## 1️⃣ Data Preprocessing
- Image resizing (128 × 128)
- Conversion from BGR to HSV color space
- Standardization of image format

---

## 2️⃣ Feature Engineering
Extracted handcrafted features:

- Color Features → HSV Histogram  
- Texture Features → Local Binary Pattern (LBP)  
- Shape Features → Hu Moments  

Total Features Extracted: 801 per image  

---

## 3️⃣ Feature Selection
- Method: SelectKBest (Mutual Information)
- Reduced features: 801 → 100

Benefits:
- Removes irrelevant features  
- Improves accuracy  
- Reduces computation time  

---

## 4️⃣ Model Training
Models used:
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  

---

## 5️⃣ Evaluation Metrics
- Accuracy  
- F1-Score  
- 5-Fold Cross Validation  

---

## 6️⃣ Visualizations
- Model Comparison (SVM vs KNN vs Random Forest)
- Feature Comparison (Individual vs Combined features)
- Confusion Matrix for all models  

---

# 📈 Results Summary

| Model | Accuracy |
|------|--------|
| SVM | Lower |
| KNN | Moderate |
| Random Forest | ⭐ 83.5% (Best) |

---

# 🚀 Deployment
The best-performing model is deployed using Streamlit.

### Features:
- Upload plant leaf image  
- Real-time prediction  
- Displays disease or healthy output  

---

# 🌐 Live Demo  
https://predictiveproject1-x6cg8tzb5uwncmiqdz2nys.streamlit.app/

---

