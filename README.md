# PredictiveProject1
Plant Disease Classification Using Machine Learning
# Plant Disease Classification

## Problem Statement
Plant diseases affect the crop quality and it also reduces agricultural productivity.
Farmers often find it difficult to identify the diseases at early stage which causes problems in the agricultural sector. 
This project aims to build a machine learning model that can be used to detect the plant diseases from leaf images and it helps to classify them as healthy or diseased.

## Dataset
In this project the dataset used is PlantVillage dataset.
It contains images of plant leaves categorized into different classes such as:
- Healthy
- Bacterial Spot
- Early Blight
- Late Blight

The dataset consists of labeled images, which are used to train and test 
the machine learning model.

## Steps
- Data preprocessing
- Feature engineering & selection (color and texture-based features)
- Model training
- Evaluation
- Visualizations
 ## 1. Model Comparison (SVM vs KNN vs Random Forest)
Grouped bar chart comparing Accuracy, F1-Score, and 5-Fold CV Accuracy for all three models
## 2. Individual vs Combined Features
Bar chart showing Random Forest accuracy using Color Histogram, LBP, Hu Moments, and Combined features separately
## 3. Confusion Matrix
Side-by-side confusion matrices for SVM, KNN, and Random Forest using a blue colormap


## Accuracy
Random Forest is the best model and it has the accuracy of 83.5%

## Deployment
The model is deployed using Streamlit, allowing users to upload leaf images and get predictions.

## Streamlit App
https://predictiveproject1-x6cg8tzb5uwncmiqdz2nys.streamlit.app/
