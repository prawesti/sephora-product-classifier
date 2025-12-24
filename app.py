import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from pytorch_tabnet.tab_model import TabNetClassifier

# --- 1. CONFIG HALAMAN ---
st.set_page_config(
    page_title="Sephora AI Predictor",
    page_icon="💄",
    layout="wide"
)

# --- 2. CUSTOM CSS UNTUK TAMPILAN MODERN ---
st.markdown("""
    <style>
    /* Mengubah font dan background */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Desain Card untuk Hasil */
    .prediction-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #ff4b4b;
        margin-top: 20px;
    }
    
    /* Tombol Kustom */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e03e3e;
        border: none;
        color: white;
    }
    
    /* Judul Sidebar */
    .css-1639199 { 
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_tabular_resources():
    try:
        m1 = load_model('model_mlp.h5')
        m2 = TabNetClassifier()
        m2.load_model('model_tabnet.zip')
        m3 = load_model('model_embedding.h5')
        with open('mlp_artifacts.pkl', 'rb') as f:
            art = pickle.load(f)
        return m1, m2, m3, art
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

m1, m2, m3, art = load_tabular_resources()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("img.jpg", width=200)
    st.markdown("---")
    st.header("Konfigurasi Model")
    pilihan = st.selectbox(
        "Pilih Arsitektur:", 
        ["MLP (Base)", "TabNet (Pretrained 1)", "Embedding + NN (Pretrained 2)"]
    )
    st.markdown("---")
    st.info("Aplikasi ini menggunakan kecerdasan buatan untuk menentukan kategori produk Sephora berdasarkan data tabular.")

# --- 5. MAIN UI ---
st.title("✨ Sephora Product Category Predictor")
st.markdown("Masukkan spesifikasi produk di bawah ini untuk mendapatkan prediksi kategori yang akurat.")

# Menggunakan Kolom untuk Form Input
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛍️ Identitas Produk")
        brand = st.selectbox("Brand Name", art['le_brand'].classes_)
        price = st.number_input("Price (USD)", min_value=0.0, value=20.0, step=0.5)
        
    with col2:
        st.subheader("📊 Statistik & Popularitas")
        rating = st.slider("Rating Produk", 0.0, 5.0, 4.0, step=0.1)
        loves = st.number_input("Loves Count", min_value=0, value=100)
        reviews = st.number_input("Reviews Count", min_value=0, value=10)

st.markdown("<br>", unsafe_allow_html=True)

# Tombol Prediksi
predict_btn = st.button("🚀 Jalankan Prediksi")

# --- 6. LOGIKA PREDIKSI ---
if predict_btn:
    if art is not None:
        # Preprocessing
        brand_enc = art['le_brand'].transform([brand])[0]
        num_data = np.array([[loves, rating, reviews, price]])
        full_data = np.array([[brand_enc, loves, rating, reviews, price]])
        
        with st.spinner('Menganalisis data...'):
            if pilihan == "MLP (Base)":
                scaled = art['scaler'].transform(full_data)
                res = m1.predict(scaled)
                idx = np.argmax(res)
            elif pilihan == "TabNet (Pretrained 1)":
                idx = m2.predict(full_data)[0]
            else: # Embedding + NN
                res = m3.predict([np.array([brand_enc]), num_data])
                idx = np.argmax(res)
            
            final_label = art['le_target'].inverse_transform([idx])[0]

        # Tampilan Hasil Prediksi
        st.markdown(f"""
            <div class="prediction-card">
                <p style="color: #666; font-size: 18px; margin-bottom: 5px;">Kategori Produk Terdeteksi:</p>
                <h1 style="color: #ff4b4b; margin-top: 0;">{final_label}</h1>
                <p style="font-style: italic; color: #888;">Dianalisis menggunakan {pilihan}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model tidak tersedia. Pastikan file .h5, .zip, dan .pkl sudah ada.")

# --- 7. FOOTER ---
st.markdown("<br><hr><center><small>UAP Pembelajaran Mesin - Sephora Classifier Project</small></center>", unsafe_allow_html=True)