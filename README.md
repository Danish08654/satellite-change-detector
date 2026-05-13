# 🛰️ Satellite Temporal Change Detector

An AI-powered satellite image change detection system that analyses temporal satellite image pairs to identify environmental and land-use changes.
The project combines a Siamese EfficientNet CNN, XGBoost feature engineering, FastAPI backend, and Streamlit dashboard to detect:

- 🌲 Deforestation
- 🌊 Floods
- 🏗️ Construction / Urban Expansion
- 🌾 Agricultural Changes
- ✅ No Significant Change

---

##  Features

- Upload Before & After satellite images
- AI-based temporal change detection
- Pixel-level change heatmap
- NDVI & water index analysis
- Interactive Streamlit dashboard
- FastAPI REST API backend
- Ensemble prediction system
- Global change visualization

---

##  Model Architecture

### 1. Siamese EfficientNet-B0
Processes before and after satellite images through a shared CNN backbone to learn temporal visual differences.

### 2. XGBoost Feature Classifier
Uses engineered features including:
- Pixel difference statistics
- NDVI vegetation change
- Water index variation
- Texture and edge changes

### 3. Ensemble Prediction
Final prediction combines:
- 50% Siamese CNN prediction
- 50% XGBoost prediction

---


 How to Test

Upload a Before satellite image
Upload an After satellite image
Click Detect Changes


The system will display:


Predicted change type
Confidence score
Severity level
Change heatmap
Vegetation & water index changes

---

 Technologies Used
 
Python
PyTorch
EfficientNet
XGBoost
OpenCV
FastAPI
Streamlit
Plotly
NumPy
PIL

---

Future Improvements

Real Sentinel-2 satellite integration
Google Earth Engine support
Segmentation-based change localization
Cloud deployment
Real-time monitoring dashboard

---

Author

Danish Zulfiqar


├── requirements.txt
└── README.md
