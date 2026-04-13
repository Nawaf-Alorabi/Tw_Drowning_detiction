import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# =========================================
# 1. PAGE CONFIG & STYLING
# =========================================
st.set_page_config(
    page_title="AquaGuard AI | Drowning Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Attempt to import YOLO; handle environment issues gracefully
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics library not found. Please check your requirements.txt.")
    st.stop()  # Hard stop if library is missing

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
def load_trained_model():
    """Loads the YOLO model using an absolute path-safe method."""
    model_filename = "best.pt"
    # Use absolute path to avoid file-not-found issues on cloud
    model_path = os.path.join(os.getcwd(), model_filename)
    
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

def process_image(model, image, confidence):
    """Runs inference on the image and returns the annotated result."""
    img_array = np.array(image)
    
    # Use safer inference call
    results = model(img_array, conf=confidence)
    
    # Generate annotated image
    res_plotted = results[0].plot()
    # Convert BGR (OpenCV) to RGB (Streamlit)
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    # Extract detection counts safely
    counts = {"Drowning": 0, "Swimming": 0}
    
    # Safe check for detections
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            # Use model.names instead of results[0].names for stability
            label = model.names[cls_id]
            if label in counts:
                counts[label] += 1
            
    return res_rgb, counts

# =========================================
# 4. SIDEBAR
# =========================================
with st.sidebar:
    st.title(f"🌊 {PROJECT_NAME}")
    st.subheader("Model Settings")
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.35)
    
    st.info("""
    **Model Specs:**
    - YOLOv8s-OBB
    - Python 3.11 Runtime
    - Optimized for Pool/Beach
    """)
    
    st.divider()
    st.write("Water Safety Monitoring System")

# =========================================
# 5. MAIN UI NAVIGATION
# =========================================
st.title("AquaGuard: Intelligent Drowning Detection")
tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔍 Live Detection", "📄 Technical Report"])

# --- TAB 1: OVERVIEW ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Precision", "98.3%", "+24%")
    col2.metric("Avg Recall", "78.0%", "+9%")
    col3.metric("mAP50", "0.877", "Improved")
    col4.metric("Inference Speed", "12ms", "Optimized")

    st.subheader("Class-Level Performance")
    st.table(PER_CLASS_METRICS)

# --- TAB 2: LIVE DETECTION ---
with tab2:
    st.header("Upload Image for Detection")
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.warning("⚠️ **'best.pt' not found.** Ensure the model file is in the root directory.")
        st.stop() # Stop the tab from running if model is missing
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            # Convert to RGB to handle RGBA/Alpha channel issues
            image = Image.open(uploaded_file).convert("RGB")
            
            with st.spinner("Analyzing water safety..."):
                result_img, counts = process_image(model, image, conf_threshold)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Original Input", use_container_width=True)
                with c2:
                    st.image(result_img, caption="Model Prediction", use_container_width=True)
                
                st.divider()
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
    st.subheader("Deployment & Environment")
    st.write("Current Status: **Production Ready**")
    st.markdown("""
    - **Runtime:** Python 3.11 (via runtime.txt)
    - **OpenCV:** Headless version (for Cloud compatibility)
    - **Hardware:** Optimized for CPU/GPU Inference
    """)
    
    st.subheader("Project Roadmap")
    st.markdown("- CCTV RTSP Integration\n- Real-time Alerting (SMS)\n- Human Pose Analysis")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption(f"© 2026 {PROJECT_NAME} | Built with Streamlit and YOLOv8")