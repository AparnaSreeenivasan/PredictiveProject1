%%writefile app.py

import streamlit as st
import cv2
import numpy as np
import joblib
from skimage import feature

# Page config
st.set_page_config(page_title="🌿 Plant Disease Predictor", page_icon="🌿", layout="wide")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("rf_model.pkl")
    selector = joblib.load("selector.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, selector, scaler, le

def extract_features(image):
    # Color Histograms
    hist_h = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([image], [2], None, [256], [0, 256])
    
    hist_h = cv2.normalize(hist_h, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    hist_s = cv2.normalize(hist_s, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    hist_v = cv2.normalize(hist_v, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    hist_features = np.concatenate([hist_h, hist_s, hist_v])
    
    # LBP Features
    gray = image[:, :, 2]
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = cv2.normalize(hist_lbp.astype(np.float32), None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    
    # Hu Moments
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    
    return np.concatenate([hist_features, hist_lbp, hu])

# Load models
model, selector, scaler, le = load_models()

# UI
st.title("🌿 Plant Disease Predictor")
st.markdown("Upload a leaf image to detect diseases")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            # Process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img = cv2.resize(img, (128, 128))
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Extract features
            features = extract_features(img_hsv).reshape(1, -1)
            features = selector.transform(features)
            features = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features)
            predicted_class = le.inverse_transform(prediction)[0]
            
            # Display result
            st.success(f"## 🔍 Prediction: {predicted_class}")
