import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ============================================
# VEGETABLE DATA MAPPING (Task 1.1)
# ============================================
VEGETABLE_DATA = {
    "Bean": {"emoji": "🫘", "indonesian": "Kacang"},
    "Bitter_Gourd": {"emoji": "🥒", "indonesian": "Pare"},
    "Bottle_Gourd": {"emoji": "🍐", "indonesian": "Labu Air"},
    "Brinjal": {"emoji": "🍆", "indonesian": "Terong"},
    "Broccoli": {"emoji": "🥦", "indonesian": "Brokoli"},
    "Cabbage": {"emoji": "🥬", "indonesian": "Kubis"},
    "Capsicum": {"emoji": "🫑", "indonesian": "Paprika"},
    "Carrot": {"emoji": "🥕", "indonesian": "Wortel"},
    "Cauliflower": {"emoji": "🥦", "indonesian": "Kembang Kol"},
    "Cucumber": {"emoji": "🥒", "indonesian": "Mentimun"},
    "Papaya": {"emoji": "🍈", "indonesian": "Pepaya"},
    "Potato": {"emoji": "🥔", "indonesian": "Kentang"},
    "Pumpkin": {"emoji": "🎃", "indonesian": "Labu"},
    "Radish": {"emoji": "🥕", "indonesian": "Lobak"},
    "Tomato": {"emoji": "🍅", "indonesian": "Tomat"}
}

# ============================================
# HELPER FUNCTIONS (Task 3.1, 3.3, 3.5)
# ============================================
def get_bilingual_prediction(label):
    """Return English name, Indonesian name, and emoji for a label."""
    if label in VEGETABLE_DATA:
        data = VEGETABLE_DATA[label]
        return label, data["indonesian"], data["emoji"]
    return label, label, "🥬"  # Fallback

def get_top_5_predictions(pred_array, class_names):
    """Extract top 5 predictions sorted descending by probability."""
    indices = np.argsort(pred_array)[-5:][::-1]
    return [(class_names[i], float(pred_array[i])) for i in indices]

def is_low_confidence(confidence):
    """Return True if confidence < 0.7."""
    return confidence < 0.7

# ============================================
# PAGE CONFIGURATION (Task 2.1)
# ============================================
st.set_page_config(
    page_title="Vegetable Classifier",
    page_icon="🥦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS (Task 2.1)
# ============================================
st.markdown("""
<style>
/* Main background */
.main {
    background: linear-gradient(135deg, #f0f9f0 0%, #e8f5e9 100%);
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}

/* Sidebar fix - stick to left */
[data-testid="stSidebar"] {
    left: 0;
    width: 300px;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
}
.app-header h1 {
    color: white !important;
    font-size: 2.5rem;
    margin-bottom: 10px;
}
.app-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
}
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9rem;
    margin-top: 10px;
}

/* Card styling */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Upload zone */
.upload-zone {
    border: 3px dashed #4caf50;
    border-radius: 15px;
    padding: 40px;
    text-align: center;
    background: rgba(76, 175, 80, 0.05);
    transition: all 0.3s ease;
}
.upload-zone:hover {
    background: rgba(76, 175, 80, 0.1);
    border-color: #2e7d32;
}

/* Prediction result */
.prediction-box {
    background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(46, 125, 50, 0.15);
    text-align: center;
    border: 2px solid #c8e6c9;
}
.prediction-label {
    font-size: 28px;
    font-weight: bold;
    color: #2e7d32;
    margin: 10px 0;
}
.prediction-indo {
    font-size: 20px;
    color: #558b2f;
    margin-bottom: 15px;
}
.prediction-emoji {
    font-size: 48px;
    margin-bottom: 10px;
}
.prediction-conf {
    font-size: 18px;
    color: #666;
}

/* Warning box */
.warning-box {
    background: #fff3e0;
    border: 2px solid #ff9800;
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
    color: #e65100;
}

/* Sidebar vegetable grid */
.veg-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 10px 0;
}
.veg-item {
    background: #f1f8e9;
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    font-size: 0.8rem;
}
.veg-emoji {
    font-size: 1.5rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    border-top: 1px solid #e0e0e0;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD RESOURCES
# ============================================
@st.cache_resource
def load_model():
    model_path = 'saved_model'
    if not os.path.exists(model_path):
        st.error(f"Folder model '{model_path}' tidak ditemukan.")
        return None
    try:
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_labels():
    labels_path = 'labels.txt'
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return []



# ============================================
# HEADER (Task 2.2)
# ============================================
st.markdown("""
<div class="app-header">
    <h1>🥦 Vegetable Classifier</h1>
    <p>Klasifikasi Sayuran Cerdas dengan Deep Learning</p>
    <span class="badge">🎯 15 Jenis Sayuran</span>
</div>
""", unsafe_allow_html=True)

# Grid 15 sayuran yang bisa diklasifikasi
st.markdown("#### 🥗 Sayuran yang dapat dikenali:")
cols = st.columns(5)
for idx, (label, data) in enumerate(VEGETABLE_DATA.items()):
    with cols[idx % 5]:
        display_name = label.replace("_", " ")
        st.markdown(f"""
        <div style="background:#f1f8e9;padding:10px;border-radius:10px;text-align:center;margin-bottom:10px;box-shadow:0 2px 5px rgba(0,0,0,0.1);">
            <div style="font-size:2rem;">{data['emoji']}</div>
            <div style="font-size:0.7rem;font-weight:bold;color:#2e7d32;">{display_name}</div>
            <div style="font-size:0.65rem;color:#666;">{data['indonesian']}</div>
        </div>
        """, unsafe_allow_html=True)

# Load resources
model = load_model()
class_names = load_labels()

# ============================================
# FILE UPLOADER
# ============================================
st.markdown("### 📤 Upload Gambar")
uploaded_file = st.file_uploader(
    "Drag and drop atau klik untuk memilih gambar", 
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG"
)

# ============================================
# PREDICTION DISPLAY (Task 4.3)
# ============================================
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='📷 Gambar yang diupload', use_container_width=True)
    
    with col2:
        if model:
            with st.spinner('🔍 Sedang menganalisis...'):
                # Preprocessing (same as notebook: resize -> normalize -> expand dims -> float32)
                target_size = (224, 224)
                img_resized = image.resize(target_size)
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
                input_tensor = tf.convert_to_tensor(img_array)
                
                # Inference
                try:
                    infer = model.signatures["serving_default"]
                    output = infer(input_tensor)
                    pred = list(output.values())[0].numpy()
                    pred_idx = np.argmax(pred)
                    confidence = float(np.max(pred))
                    
                    # Get bilingual prediction
                    if class_names and pred_idx < len(class_names):
                        label = class_names[pred_idx]
                        eng_name, indo_name, emoji = get_bilingual_prediction(label)
                    else:
                        eng_name, indo_name, emoji = f"Class {pred_idx}", f"Kelas {pred_idx}", "🥬"
                    
                    # Display Result Card
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-emoji">{emoji}</div>
                        <div class="prediction-label">{eng_name.replace("_", " ")}</div>
                        <div class="prediction-indo">{indo_name}</div>
                        <div class="prediction-conf">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence progress bar
                    st.write("")
                    st.write("**📊 Tingkat Kepercayaan:**")
                    st.progress(confidence)
                    
                    # Low confidence warning (Task 3.5)
                    if is_low_confidence(confidence):
                        st.markdown("""
                        <div class="warning-box">
                            ⚠️ <strong>Peringatan:</strong> Tingkat kepercayaan rendah! 
                            Coba upload gambar yang lebih jelas atau dengan pencahayaan yang lebih baik.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top 5 predictions chart (Task 3.3)
                    if class_names and len(pred[0]) == len(class_names):
                        st.write("")
                        st.write("**🏆 Top 5 Prediksi:**")
                        top_5 = get_top_5_predictions(pred[0], class_names)
                        top_5_dict = {}
                        for name, prob in top_5:
                            data = VEGETABLE_DATA.get(name, {"emoji": "🥬", "indonesian": name})
                            display = f"{data['emoji']} {name.replace('_', ' ')}"
                            top_5_dict[display] = prob
                        st.bar_chart(top_5_dict, color="#4caf50")
                        
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")
        else:
            st.error("⚠️ Model belum siap. Pastikan folder 'saved_model' tersedia.")

else:
    # Placeholder
    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:3rem;margin:0;">📷</p>
        <p style="color:#666;margin:10px 0;">Belum ada gambar yang dipilih</p>
        <p style="color:#888;font-size:0.9rem;">Silakan upload gambar sayuran untuk memulai klasifikasi</p>
        <p style="color:#aaa;font-size:0.8rem;">Format: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER (Task 6.1)
# ============================================
st.markdown("""
<div class="footer">
    <p>🥦 <strong>Vegetable Classifier</strong> v2.0.0</p>
    <p>Developed with ❤️ by <strong>Firdaus Khotibul Zickrian</strong></p>
    <p style="font-size:0.8rem;color:#999;">© 2025 - Deep Learning Vegetable Classification</p>
</div>
""", unsafe_allow_html=True)
