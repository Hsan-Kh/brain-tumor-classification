"""
app.py
NeuroGuard AI - Clinical Diagnostic Suite.
Innovative styling with Glassmorphism, AI Consensus Engine, and Dynamic Theming.
"""
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import os
import pandas as pd
import numpy as np

try:
    from src import model_builder
except ImportError:
    st.error("⚠️ Critical Error: 'src' module not found.")

# --- CONFIGURATION & ASSETS ---
st.set_page_config(
    page_title="NeuroGuard AI | Clinical Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Assets
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/5069/5069873.png" 
BANNER_URL = "https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&w=2000&q=80"

# Constants
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_DIR = "models"
DEVICE = "cpu"

# --- CSS (Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1 { color: #0f172a; font-weight: 800; letter-spacing: -1px; }
    h2, h3 { color: #334155; }
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 20px;
    }
    .badge-tumor { background-color: #fee2e2; color: #991b1b; padding: 5px 10px; border-radius: 8px; font-weight: bold; }
    .badge-safe { background-color: #dcfce7; color: #166534; padding: 5px 10px; border-radius: 8px; font-weight: bold; }
    .stFileUploader { border: 2px dashed #cbd5e1; border-radius: 15px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- CORE LOGIC ---
@st.cache_resource
def load_model(model_name):
    try:
        model = model_builder.build_model(model_name, num_classes=len(CLASS_NAMES), device=DEVICE)
        load_path = os.path.join(MODEL_DIR, f"{model_name}_brain_tumor.pth")
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path, map_location=torch.device(DEVICE)))
            model.eval()
            return model
        else:
            return None
    except Exception as e:
        return None

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, processed_image):
    with torch.no_grad():
        start_time = time.time()
        logits = model(processed_image.to(DEVICE))
        inference_time = time.time() - start_time
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        return {
            "label": CLASS_NAMES[pred_idx.item()],
            "confidence": conf.item(),
            "probabilities": probs.numpy()[0],
            "time": inference_time
        }

# --- SIDEBAR ---
st.sidebar.image(LOGO_URL, width=120)
st.sidebar.title("NeuroGuard AI")
st.sidebar.caption("Clinical Diagnostic Suite v2.1")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Operation Mode:", 
    ["Single Model Inference", "⚔️ Arena Benchmark (Consensus)"],
    captions=["Fast, single-opinion scan", "Multi-model voting system"]
)

# --- MAIN UI ---
st.image(BANNER_URL, use_container_width=True)
st.title("Patient Diagnostic Portal")
st.markdown("Upload MRI scan for real-time neural analysis.")

uploaded_file = st.file_uploader("Drop DICOM or Image file here...", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    col_img, col_data = st.columns([1, 1.5])
    
    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Scan", use_container_width=True)
        st.markdown(f"**Resolution:** {image.size}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        processed_img = process_image(image)

    with col_data:
        if st.button("Initialize Neural Scan", type="primary", use_container_width=True):
            
            # Animation
            progress_bar = st.progress(0)
            status = st.empty()
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
                if i == 30: status.text("Normalizing tensors...")
                if i == 60: status.text("Loading weights...")
                if i == 90: status.text("Running inference...")
            status.empty()
            progress_bar.empty()

            # --- MODE 1: SINGLE MODEL ---
            if mode == "Single Model Inference":
                model_name = st.sidebar.selectbox("Active Model:", ["resnet18", "mobilenet", "efficientnet", "densenet121"])
                model = load_model(model_name)
                
                if model:
                    res = predict(model, processed_img)
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Diagnostic Report")
                    
                    color_class = "badge-safe" if res['label'] == 'notumor' else "badge-tumor"
                    label_clean = res['label'].upper() if res['label'] != 'notumor' else "NO TUMOR DETECTED"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="margin:0;">{label_clean}</h2>
                        <span class="{color_class}">{res['confidence']*100:.1f}% Confidence</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
                    
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob = res['probabilities'][i]
                        bar_color = "#ef4444" if class_name == res['label'] and class_name != "notumor" else "#3b82f6"
                        if class_name == "notumor" and res['label'] == "notumor": bar_color = "#22c55e"
                        st.markdown(f"**{class_name.capitalize()}**")
                        st.progress(float(prob))
                    
                    st.caption(f"Inference Time: {res['time']:.4f}s | Model: {model_name}")
                    st.markdown('</div>', unsafe_allow_html=True)

            # --- MODE 2: ARENA CONSENSUS ---
            else:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("⚔️ Arena Consensus Engine")
                
                models = ["resnet18", "mobilenet", "efficientnet", "densenet121"]
                votes = []
                results_data = []
                
                # Dynamic columns 
                cols = st.columns(4) 
                for idx, m_name in enumerate(models):
                    model = load_model(m_name)
                    if model:
                        r = predict(model, processed_img)
                        votes.append(r['label'])
                        results_data.append(r)
                        
                        with cols[idx]:
                            st.info(f"**{m_name}**")
                            st.write(f"Pred: **{r['label']}**")
                            st.write(f"Conf: `{r['confidence']*100:.1f}%`")
                
                st.markdown("---")
                
                unique_votes = set(votes)
                if len(unique_votes) == 1:
                    winner = votes[0]
                    color = "#dcfce7" if winner == "notumor" else "#fee2e2"
                    text_color = "#166534" if winner == "notumor" else "#991b1b"
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid {text_color};">
                        <h3 style="color: {text_color}; margin:0;">✅ CONSENSUS REACHED</h3>
                        <p style="color: {text_color}; font-size: 1.2rem;">All 4 systems agree: <strong>{winner.upper()}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #ffedd5; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #c2410c;">
                        <h3 style="color: #c2410c; margin:0;">⚠️ SYSTEM CONFLICT</h3>
                        <p style="color: #c2410c;">Models disagree. Clinical review required.</p>
                        <p>Votes: {votes}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.subheader("Performance Telemetry")
                df = pd.DataFrame(results_data)
                df = df[["label", "confidence", "time"]]
                df.index = models
                st.dataframe(df.style.highlight_max(axis=0, subset=["confidence"]), use_container_width=True)

else:
    st.info("System Standby. Upload a DICOM or MRI image to initiate diagnostic protocols.")
