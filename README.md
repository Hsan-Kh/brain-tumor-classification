# 🧠 NeuroGuard AI

**An Advanced MRI Analysis & Diagnostic Suite**

NeuroGuard AI is a medical computer vision project designed to assist in the classification of Brain Tumors (Glioma, Meningioma, Pituitary) using Deep Learning. 

Unlike standard classifiers, this project features an **"Arena Consensus Engine"** that aggregates predictions from 4 different architectures to improve reliability, alongside a dedicated **EDA (Exploratory Data Analysis) Dashboard** for radiologists to inspect dataset integrity.

Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

![Project Status](https://img.shields.io/badge/Status-Prototype-blue) ![Python](https://img.shields.io/badge/Python-3.13.5-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red) ![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

---

##  English Documentation

### 🚀 Key Features

*   **Multi-Model Architecture:** Implements Transfer Learning using **ResNet18**, **MobileNetV2**, **EfficientNetB0**, and **DenseNet121** (added for superior texture analysis).
*   **The "Arena" System:** A consensus voting logic that runs all 4 models simultaneously on a single scan to cross-verify predictions.
*   **Medical Dashboard (EDA):** A standalone interface (`eda.py`) to visualize class distributions, detect outliers in image resolution, and analyze pixel intensity.
*   **Modern UI:** Built with Streamlit, featuring a custom "Glassmorphism" design and realistic medical theming.

### 🛠 Installation & Usage

1.  **Clone the repo**
    ```bash
    git clone https://github.com/Hsan-Kh/brain-tumor-classification.git
    cd brain-tumor-classification
    cd NeuroGuardAI
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Applications**
    *   For the Diagnostic Tool:
        ```bash
        streamlit run app.py
        ```
    *   For the Analysis Dashboard:
        ```bash
        streamlit run eda.py
        ```

4.  **Training **
    If you want to train the models:
    ```bash
    python train_arena.py
    ```

### 📂 Project Structure

*   `app.py`: Main inference interface for users/doctors.
*   `eda.py`: Dashboard for data visualization and model architecture explanation.
*   `src/`: Contains the engine for training (`engine.py`) and model construction (`model_builder.py`).
*   `train_arena.py`: Script to train and save models to `.pth` files.

---

##  Documentation Française

### 💡 À propos du projet

NeuroGuard AI est un outil de vision par ordinateur médicale conçu pour classifier les tumeurs cérébrales (Gliome, Méningiome, Tumeur Hypophysaire) à partir d'IRM.

L'objectif n'est pas seulement de prédire, mais de **sécuriser la décision** via un système de "Consensus" qui compare les avis de 4 modèles différents, et d'offrir une transparence totale sur les données via un tableau de bord analytique.

### 🌟 Fonctionnalités Clés

*   **Architecture Multi-Modèles :** Utilisation du Transfer Learning sur **ResNet18** (référence), **MobileNetV2** (léger), **EfficientNetB0** (précis) et **DenseNet121** (optimisé pour les textures médicales).
*   **Système "Arena" :** Un moteur de vote qui agrège les prédictions des 4 modèles pour fournir un score de confiance unifié.
*   **Tableau de Bord EDA :** Une interface dédiée (`eda.py`) pour visualiser la distribution des classes, détecter les anomalies (outliers) et inspecter la qualité des images.
*   **Interface Moderne :** Application Streamlit avec un design "Glassmorphism" et une ergonomie adaptée au contexte médical.

### ⚙️ Installation et Utilisation

1.  **Cloner le projet**
    ```bash
    git clone https://github.com/Hsan-Kh/brain-tumor-classification.git
    cd brain-tumor-classification
    cd NeuroGuardAI
    ```

2.  **Installer les dépendances**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Lancer les Applications**
    *   Pour l'outil de diagnostic clinique :
        ```bash
        streamlit run app.py
        ```
    *   Pour le tableau de bord d'analyse :
        ```bash
        streamlit run eda.py
        ```

4.  **Entraînement **
    Si vous souhaitez entraîner les modèles :
    ```bash
    python train_arena.py
    ```

### 📂 Structure du Projet

*   `app.py` : Interface principale d'inférence (diagnostic).
*   `eda.py` : Tableau de bord pour la visualisation des données et l'explication des architectures.
*   `src/` : Contient le moteur d'entraînement (`engine.py`) et la construction des modèles (`model_builder.py`).
*   `train_arena.py` : Script pour entraîner les modèles et sauvegarder les fichiers de poids (`.pth`).


---

## ⚖️ License/Licence
Distributed under the MIT License. See `LICENSE` for more information.  /  Distribué sous Licence MIT. Voir `LICENSE` pour plus d'informations.


---
**Author / Auteur :** Hsan KHECHAREM
