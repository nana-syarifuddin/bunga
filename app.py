import sys
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.header("Informasi Lingkungan Python")
st.write(f"Versi Python yang digunakan: **{sys.version}**")
st.write(f"Major.Minor Python: **{sys.version_info.major}.{sys.version_info.minor}**")

try:
    import tensorflow as tf
    st.write(f"Versi TensorFlow terinstal: **{tf.__version__}**")
except ImportError:
    st.error("TensorFlow TIDAK terinstal. Ada masalah dependensi.")

# Load model yang sudah dilatih
model = tf.keras.models.load_model('models/best_model.h5')
print("Model input shape:", model.input_shape)

# Daftar kelas bunga (ganti jika berbeda)
class_names = ['daisy', 'dandelion', 'lily', 'orchid', 'rose', 'sunflower', 'tulip']

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    image = image.resize((150, 150))  # Ukuran input saat pelatihan model
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Tambahkan batch dimension
    img_array = preprocess_input(img_array)    # Preprocessing khusus MobileNetV2
    return img_array


# Judul aplikasi
st.title("🌸 Prediksi Gambar Bunga")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar bunga (jpg/jpeg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    st.write("🔍 Memprediksi...")
    img = preprocess_image(image)
    prediction = model.predict(img)

    st.write("Shape hasil prediksi:", prediction.shape) 
    st.write("Isi prediksi:", prediction)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"🌼 Prediksi: **{predicted_class}**")
    st.info(f"📊 Akurasi keyakinan: **{confidence*100:.2f}%**")
