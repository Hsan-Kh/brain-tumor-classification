"""
eda.py
NeuroGuard AI - Professional Medical Dashboard.
Enhanced with realistic themes, medical imagery, and custom styling.
"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import random
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

try:
    from src import model_builder
except ImportError:
    st.error("Error: Could not import 'src.model_builder'. Make sure this file is in the root directory.")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroGuard AI | Medical Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS & PATHS ---
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODEL_DIR = "models"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
DEVICE = "cpu"

# --- ASSETS ---
BANNER_URL = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=2000&q=80" 
SIDEBAR_LOGO = "https://cdn-icons-png.flaticon.com/512/5069/5069873.png"
MRI_SCANNER_IMG = "https://images.unsplash.com/photo-1516549655169-df83a0774514?auto=format&fit=crop&w=1000&q=80"
CODE_IMG = "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb?auto=format&fit=crop&w=1000&q=80"
NETWORK_IMG = "https://images.unsplash.com/photo-1507413245164-6160d8298b31?auto=format&fit=crop&w=1000&q=80"

# --- CSS ---
st.markdown("""
    <style>
    .main .block-container { max-width: 95%; padding: 1rem; }
    h1 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    h2 { color: #334155; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; font-family: 'Segoe UI', sans-serif; }
    h3 { color: #475569; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #1e293b; }
    div[data-testid="stGraphVizChart"] { display: flex; justify-content: center; }
    .medical-banner { border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
    
    /* TAB STYLING */
    div[data-baseweb="tab-list"] { gap: 8px; width: 100%; }
    button[data-baseweb="tab"] { flex-grow: 1 !important; height: 60px !important; background-color: #f1f5f9; border-radius: 8px 8px 0px 0px; }
    button[data-baseweb="tab"] div p { font-size: 22px !important; font-weight: 700 !important; }
    button[aria-selected="true"] { background-color: #ffffff !important; border-top: 4px solid #3b82f6 !important; box-shadow: 0 -2px 5px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_dataset_stats():
    data = []
    for split in ["Training", "Testing"]:
        path = os.path.join(DATA_DIR, split)
        if not os.path.exists(path):
            continue 
        for label in CLASSES:
            folder_path = os.path.join(path, label)
            files = glob.glob(os.path.join(folder_path, "*"))
            data.append({"Split": split, "Class": label, "Count": len(files), "Files": files})
    return pd.DataFrame(data)

@st.cache_resource
def load_trained_model(model_name):
    try:
        model = model_builder.build_model(model_name, num_classes=len(CLASSES), device=DEVICE)
        weight_path = os.path.join(MODEL_DIR, f"{model_name}_brain_tumor.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            model.eval()
            return model
        else:
            return None
    except Exception as e:
        return None

def get_image_dimensions_sample(files, sample_size=100):
    dims = []
    sample = random.sample(files, min(len(files), sample_size))
    for f in sample:
        try:
            with Image.open(f) as img:
                dims.append(img.size)
        except:
            pass
    return pd.DataFrame(dims, columns=["Width", "Height"])

# --- SIDEBAR ---
st.sidebar.image(SIDEBAR_LOGO, width=120)
st.sidebar.title("NeuroGuard AI")
st.sidebar.info("**Version:** 2.1.0 (Medical Edition)")
st.sidebar.markdown("---")
st.sidebar.subheader("Clinical Context")
st.sidebar.caption("""
This dashboard supports radiologists in verifying dataset integrity and evaluating model reliability for:
- Gliomas
- Meningiomas
- Pituitary Tumors
""")

# --- MAIN HERO ---
st.image(BANNER_URL, use_container_width=True)
st.title("NeuroGuard AI: Analytics Dashboard")
st.markdown("**System Status:** Ready | **Dataset:** MRI Brain Tumor (Kaggle)")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 Data Explorer", 
    "🧬 Preprocessing", 
    "🧠 Architectures",
    "📊 Performance Report"
])

# ==============================================================================
# TAB 1: DATA EXPLORATION
# ==============================================================================
with tab1:
    col_head1, col_head2 = st.columns([1, 3])
    with col_head1:
        st.image(MRI_SCANNER_IMG, use_container_width=True)
    with col_head2:
        st.header("1. Dataset Cohort Analysis")
        st.markdown("Statistical overview of the patient cohort and image distribution.")

    df_stats = load_dataset_stats()
    
    if df_stats.empty:
        st.error("Data directory not found.")
    else:
        # 1.1 Global Stats
        col1, col2, col3 = st.columns(3)
        total_train = df_stats[df_stats["Split"]=="Training"]["Count"].sum()
        total_test = df_stats[df_stats["Split"]=="Testing"]["Count"].sum()
        col1.metric("Training Cohort", f"{total_train} Scans")
        col2.metric("Validation Cohort", f"{total_test} Scans")
        col3.metric("Pathology Classes", len(CLASSES))

        # 1.2 Seaborn
        st.subheader("A. Pathology Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=df_stats, x="Class", y="Count", hue="Split", palette="mako", ax=ax)
        ax.set_title("Count of Scans per Pathology")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig)

        # 1.3 Outlier
        st.subheader("B. Scan Dimensions Quality Control")
        all_train_files = [f for row in df_stats[df_stats["Split"]=="Training"]["Files"] for f in row]
        df_dims = get_image_dimensions_sample(all_train_files, sample_size=200)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df_dims, x="Width", y="Height", alpha=0.6, color="#e11d48", ax=ax2)
        ax2.set_title("Resolution Consistency Check")
        st.pyplot(fig2)
        
        # 1.4 Pixel Intensity
        st.subheader("C. Radiodensity Distribution")
        if st.button("Run Density Analysis"):
            sample_files = random.sample(all_train_files, min(len(all_train_files), 20))
            pixel_vals = np.concatenate([np.array(Image.open(f).convert('L')).flatten() for f in sample_files])
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            sns.histplot(pixel_vals, bins=50, color="#3b82f6", kde=True, stat="density", ax=ax3)
            ax3.set_title("Voxel Intensity Histogram")
            st.pyplot(fig3)

# ==============================================================================
# TAB 2: PREPROCESSING
# ==============================================================================
with tab2:
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        st.image(CODE_IMG, use_container_width=True)
    with col_p2:
        st.header("2. Image Preprocessing Pipeline")
        st.markdown("Standardization protocols applied to raw DICOM/JPG exports.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Protocol Definition")
        st.code("""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
        """, language="python")
    
    with c2:
        st.subheader("Deep Feature Extraction")
        st.info("Why manual feature selection is not used:")
        st.markdown("""
        In modern Medical AI, **Convolutional Neural Networks (CNNs)** replace manual feature engineering.
        The network learns hierarchical features automatically:
        1.  **Low Level:** Edges & Gradients.
        2.  **Mid Level:** Tissue Textures & Shapes.
        3.  **High Level:** Tumor Pathology specific patterns.
        """)

    st.subheader("Visual Verification")
    if not df_stats.empty:
        file_list = df_stats.iloc[0]["Files"]
        if file_list:
            img_path = random.choice(file_list)
            raw_img = Image.open(img_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            proc_tensor = transform(raw_img)
            proc_img_view = proc_tensor.permute(1, 2, 0).numpy()
            proc_img_view = (proc_img_view * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            proc_img_view = np.clip(proc_img_view, 0, 1)

            ic1, ic2 = st.columns(2)
            ic1.image(raw_img, caption="Raw Input Scan", use_container_width=True)
            ic2.image(proc_img_view, caption="Normalized Tensor Input", use_container_width=True)

# ==============================================================================
# TAB 3: MODEL ARCHITECTURES 
# ==============================================================================
with tab3:
    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        st.image(NETWORK_IMG, use_container_width=True)
    with col_m2:
        st.header("3. Neural Architectures")
        st.markdown("Comparative analysis of the backbone networks.")

    st.markdown("---")

    # --- 1. RESNET 18 ---
    st.subheader("1. ResNet18 (The Reliable Standard)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.graphviz_chart('''
            digraph {
                rankdir=LR;
                node [shape=box style=filled fillcolor="#e0f2fe" fontname="Helvetica"];
                Input [shape=oval];
                Conv1; Relu [shape=circle width=0.5 fixedsize=true]; Conv2;
                Add [label="+" shape=circle width=0.5 fixedsize=true fillcolor="#facc15"];
                Output [shape=oval];
                Input -> Conv1 -> Relu -> Conv2 -> Add;
                Input -> Add [label="Skip" style=dashed color="red"];
                Add -> Output;
            }
        ''', use_container_width=True)
    with c2:
        st.markdown("""
        **Architecture:** Residual Network (18 Layers)
        **Key Innovation:** **Skip Connections**.
        **How it works:** Introduces "identity shortcut connections" that skip layers to solve vanishing gradients.
        """)

    st.divider()

    # --- 2. MOBILENET V2 ---
    st.subheader("2. MobileNet V2 (Optimized for Efficiency)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.graphviz_chart('''
            digraph {
                rankdir=LR;
                node [shape=box style=filled fillcolor="#f0fdf4" fontname="Helvetica"];
                Input [shape=oval];
                subgraph cluster_0 { label="Standard Conv"; style=filled; color=lightgrey; BigConv [label="Heavy 3x3"]; }
                subgraph cluster_1 { label="MobileNet Split"; style=filled; color="#ecfccb"; Depth [label="Depthwise"]; Point [label="Pointwise"]; }
                Input -> BigConv [style=invis]; Input -> Depth -> Point;
            }
        ''', use_container_width=True)
    with c2:
        st.markdown("""
        **Architecture:** Lightweight CNN
        **Key Innovation:** **Depthwise Separable Convolutions**.
        **How it works:** Splits convolution into Depthwise (Filtering) and Pointwise (Combining), reducing parameters by ~9x.
        """)

    st.divider()

    # --- 3. EFFICIENTNET ---
    st.subheader("3. EfficientNet B0 (State-of-the-Art Scaling)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.graphviz_chart('''
            digraph {
                rankdir=LR;
                node [shape=box style=filled fillcolor="#fff7ed" fontname="Helvetica"];
                Base [label="Base Model"];
                Scale [label="Compound\\nScaling" shape=diamond fillcolor="#facc15"];
                Width; Depth; Res;
                Base -> Scale; Scale -> Width; Scale -> Depth; Scale -> Res;
            }
        ''', use_container_width=True)
    with c2:
        st.markdown("""
        **Architecture:** Compound Scaled CNN
        **Key Innovation:** **Compound Scaling Method**.
        **How it works:** Uniformly scales network width, depth, and resolution based on a compound coefficient.
        """)
    
    st.divider()

    # --- 4. DENSENET 121 ---
    st.subheader("4. DenseNet121 (Feature Reuse Powerhouse)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.graphviz_chart('''
            digraph {
                rankdir=LR;
                node [shape=box style=filled fillcolor="#f3e8ff" fontname="Helvetica"];
                Input [shape=oval];
                L1 [label="Layer 1"];
                L2 [label="Layer 2"];
                L3 [label="Layer 3"];
                Output [shape=oval];

                Input -> L1;
                Input -> L2 [color="purple"];
                Input -> L3 [color="purple"];
                
                L1 -> L2;
                L1 -> L3 [color="purple"];
                
                L2 -> L3;
                L3 -> Output;
            }
        ''', use_container_width=True)
    with c2:
        st.markdown("""
        **Architecture:** Densely Connected Network
        **Key Innovation:** **Feature Concatenation**.
        **How it works:** Unlike ResNet which *adds* numbers, DenseNet **stacks** them. Every layer receives inputs from ALL previous layers.
        **Why for MRI?** It forces the model to "remember" low-level tumor textures from the start, preventing feature loss.
        """)

# ==============================================================================
# TAB 4: PERFORMANCE
# ==============================================================================
with tab4:
    st.header("4. Clinical Performance Evaluation")
    st.markdown("Evaluation metrics on the independent Test Set.")
    
    model_choice = st.selectbox("Select Model for Report", ["resnet18", "mobilenet", "efficientnet", "densenet121"])

    @st.cache_data(show_spinner="Running inference on full Test Set... (This happens once)")
    def get_predictions(model_name):
        model = load_trained_model(model_name)
        if model is None: return None, None
        y_true_cache, y_pred_cache, test_files = [], [], []
        
        for label in CLASSES:
            path = os.path.join(TEST_DIR, label)
            if os.path.exists(path):
                files = glob.glob(os.path.join(path, "*"))
                for f in files: test_files.append((f, CLASSES.index(label)))
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        model.eval()
        with torch.no_grad():
            for fpath, label_idx in test_files:
                try:
                    img = Image.open(fpath).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(DEVICE)
                    outputs = model(tensor)
                    _, preds = torch.max(outputs, 1)
                    y_true_cache.append(label_idx)
                    y_pred_cache.append(preds.item())
                except Exception: continue
                    
        return y_true_cache, y_pred_cache

    if st.button(f"Generate Report for {model_choice}"):
        y_true, y_pred = get_predictions(model_choice)
        
        if y_true is None:
            st.error(f"Could not load {model_choice}. Please check 'models/' folder.")
        else:
            acc = accuracy_score(y_true, y_pred)
            st.metric("Global Accuracy", f"{acc*100:.2f}%", delta="Model Performance")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, ax=ax_cm)
                st.pyplot(fig_cm)
            with col_b:
                st.subheader("Detailed Classification Report")
                report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.highlight_max(axis=0, color='#dcfce7'))
