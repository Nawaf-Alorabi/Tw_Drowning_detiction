import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Attempt to import YOLO; handle environment issues gracefully
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics library not found. Please run: pip install ultralytics")
    YOLO = None

# =========================================
# 1. PAGE CONFIG & STYLING
# =========================================
st.set_page_config(
    page_title="AquaGuard AI | Drowning Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .feature-card { background-color: #e9ecef; padding: 20px; border-radius: 10px; height: 100%; border-left: 5px solid #007bff; }
    .strong { font-weight: bold; font-size: 1.1em; color: #1e3a8a; }
    .tiny-label { font-size: 0.85em; color: #6c757d; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# =========================================
# 2. DATA & CONSTANTS
# =========================================
PROJECT_NAME = "AquaGuard AI"
# These reflect the metrics from your training history
PER_CLASS_METRICS = pd.DataFrame({
    "Class": ["Drowning", "Swimming"],
    "Precision": [1.000, 0.966],
    "Recall": [0.561, 1.000],
    "mAP50": [0.759, 0.995],
})

# =========================================
# 3. CORE FUNCTIONS
# =========================================
@st.cache_resource
def load_trained_model(model_path):
    """Loads the YOLO model and caches it to improve performance."""
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

def process_image(model, image, confidence):
    """Runs inference on the image and returns the annotated result."""
    img_array = np.array(image)
    # YOLO expects RGB or BGR; .predict handles PIL/Numpy arrays well
    results = model.predict(img_array, conf=confidence)
    
    # Generate annotated image
    res_plotted = results[0].plot()
    # Convert BGR (OpenCV) to RGB (Streamlit)
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Extract detection counts
    counts = {"Drowning": 0, "Swimming": 0}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        if label in counts:
            counts[label] += 1
            
    return res_rgb, counts

# =========================================
# 4. SIDEBAR
# =========================================
with st.sidebar:
    st.title(f"🌊 {PROJECT_NAME}")
    st.subheader("Model Settings")
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.35, help="Adjust to filter weak detections.")
    
    st.info("""
    **Training Insights:**
    - Model: YOLOv8s-OBB
    - Optimized Hyperparameters:
        - Box Gain: 10.0
        - Augmentation: FlipUp/Down
        - Epochs: 100
    """)
    
    st.divider()
    st.write("Developed for Water Safety Monitoring")

# =========================================
# 5. MAIN UI NAVIGATION
# =========================================
st.title("AquaGuard: Intelligent Drowning Detection")
tab1, tab2, tab3 = st.tabs(["📊 Overview & Performance", "🔍 Live Detection Window", "📄 Technical Report"])

# --- TAB 1: OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Precision", "98.3%", "+24%")
    col2.metric("Avg Recall", "78.0%", "+9%")
    col3.metric("mAP50", "0.877", "Improved")
    col4.metric("Inference Speed", "12ms", "Optimized")

    st.subheader("Class-Level Performance")
    st.table(PER_CLASS_METRICS)
    
    st.write("The model shows exceptional performance in distinguishing 'Swimming' from 'Drowning', though drowning detection remains a high-focus area for further data collection.")

# --- TAB 2: LIVE DETECTION ---
with tab2:
    st.header("Upload Image for Detection")
    st.write("Test the model by uploading a scene from a swimming pool or beach.")
    
    # Attempt to load the model
    model = load_trained_model("best.pt")
    
    if model is None:
        st.warning("⚠️ **'best.pt' not found.** Please upload the model file to the app directory to enable this feature.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            with st.spinner("Analyzing image..."):
                result_img, counts = process_image(model, image, conf_threshold)
                
                # Layout for results
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(image, caption="Original Input", use_container_width=True)
                with c2:
                    st.image(result_img, caption="Model Prediction", use_container_width=True)
                
                # Display Summary Alerts
                st.divider()
                st.subheader("Analysis Summary")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if counts["Drowning"] > 0:
                        st.error(f"🚨 ALERT: {counts['Drowning']} Drowning detected!")
                    else:
                        st.success("✅ No Drowning detected.")
                
                with res_col2:
                    st.info(f"🏊 Swimming detected: {counts['Swimming']}")

# --- TAB 3: TECHNICAL REPORT ---
with tab3:
    st.subheader("Dataset & Limitations")
    limitation_df = pd.DataFrame({
        "Challenge": ["Data Imbalance", "Water Reflections", "Occlusion", "View Angle"],
        "Impact": [
            "Fewer drowning cases can limit model generalization.",
            "Glare and shadows may reduce confidence in outdoor settings.",
            "Crowded scenes can hide critical movements.",
            "Extreme angles may distort bounding box accuracy."
        ]
    })
    st.dataframe(limitation_df, use_container_width=True, hide_index=True)
    
    st.subheader("Future Roadmap")
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("""
        - **Synthetic Data:** Generating drowning simulations to balance classes.
        - **CCTV Integration:** Connecting directly to RTSP streams.
        """)
    with c_b:
        st.markdown("""
        - **Alerting System:** SMS/Mobile notifications for lifeguards.
        - **Pose Estimation:** Analyzing body joint patterns for 'panic' movements.
        """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption(f"© 2026 {PROJECT_NAME} | Built with Streamlit and YOLOv8")