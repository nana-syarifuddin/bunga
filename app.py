import sys
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os # Untuk memeriksa keberadaan file
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Bagian 1: Informasi Lingkungan & Debugging ---
st.header("Informasi Lingkungan Deployment")
st.write(f"Versi Python yang digunakan: **{sys.version}**")
st.write(f"Major.Minor Python: **{sys.version_info.major}.{sys.version_info.minor}**")

try:
    # Memastikan TensorFlow benar-benar bisa diimpor
    import tensorflow as tf
    st.write(f"Versi TensorFlow terinstal: **{tf.__version__}**")
except ImportError:
    st.error("❌ TensorFlow TIDAK terinstal. Ada masalah dependensi.")
    st.stop() # Hentikan eksekusi jika TensorFlow tidak ada

# --- Bagian 2: Definisi Custom Objects (SANGAT PENTING JIKA MODEL ANDA MEMILIKINYA) ---
# Jika model Anda menggunakan custom layer, fungsi aktivasi, atau loss function,
# Anda HARUS mendefinisikannya di sini agar Keras bisa merekonstruksinya.
# Jika Anda yakin model Anda TIDAK memiliki custom object, Anda bisa mengosongkan dictionary custom_objects.

custom_objects = {
    # Contoh custom layer:
    # 'NamaCustomLayerAnda': NamaCustomLayerAnda, # Ganti dengan nama kelas layer kustom Anda
    # class NamaCustomLayerAnda(tf.keras.layers.Layer):
    #     def __init__(self, units, **kwargs):
    #         super(NamaCustomLayerAnda, self).__init__(**kwargs)
    #         self.units = units
    #         # ... inisialisasi layer Anda
    #     def call(self, inputs):
    #         # ... logika forward pass Anda
    #         return inputs
    #     def get_config(self):
    #         config = super(NamaCustomLayerAnda, self).get_config()
    #         config.update({"units": self.units})
    #         return config

    # Contoh custom activation function:
    # 'custom_mish': lambda x: x * tf.math.tanh(tf.math.softplus(x)),
    # Atau jika didefinisikan sebagai fungsi:
    # def custom_mish(x):
    #     return x * tf.math.tanh(tf.math.softplus(x))
    # 'custom_mish': custom_mish,

    # Tambahkan semua custom object lain yang model Anda gunakan di sini
}

# --- Bagian 3: Memuat Model ---
model_path = 'models/best_model.h5'

# Pastikan file model ada sebelum mencoba memuatnya
if not os.path.exists(model_path):
    st.error(f"❌ Error: File model tidak ditemukan di `{model_path}`.")
    st.info("Pastikan file `best_model.h5` ada di dalam folder `models` di root repositori Anda.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan

try:
    # Muat model dengan mempertimbangkan custom objects
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    st.success("✅ Model berhasil dimuat!")
    st.write(f"Shape input model: `{model.input_shape}`")
except Exception as e:
    st.error("⚠️ **Gagal memuat model!**")
    st.write("Ini bisa disebabkan oleh ketidakcocokan versi TensorFlow/Keras saat menyimpan dan memuat model, atau masalah dengan custom objects.")
    st.exception(e) # Menampilkan detail traceback error di UI Streamlit
    st.stop() # Hentikan eksekusi aplikasi jika model gagal dimuat

# --- Bagian 4: Definisi Aplikasi Utama ---

# Daftar kelas bunga (Ganti sesuai dengan urutan kelas saat pelatihan model Anda)
class_names = ['daisy', 'dandelion', 'lily', 'orchid', 'rose', 'sunflower', 'tulip']

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    # Pastikan ukuran sesuai dengan input model Anda (misal 150x150)
    image = image.resize((model.input_shape[1], model.input_shape[2]))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Tambahkan batch dimension (1, height, width, channels)
    img_array = preprocess_input(img_array) # Preprocessing khusus MobileNetV2
    return img_array

# Judul aplikasi Streamlit
st.title("🌸 Prediksi Gambar Bunga")
st.markdown("Unggah gambar bunga dan biarkan AI kami memprediksinya!")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar bunga (jpg/jpeg/png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    st.write("---")
    st.write("🔍 Memprediksi gambar...")

    try:
        img_processed = preprocess_image(image)
        prediction = model.predict(img_processed)

        # Debugging: Tampilkan shape dan isi prediksi mentah
        # st.write("Shape hasil prediksi:", prediction.shape)
        # st.write("Isi prediksi (raw):", prediction)

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(prediction) * 100 # Konversi ke persentase

        st.write("---")
        st.success(f"🌼 Prediksi: **{predicted_class}**")
        st.info(f"📊 Tingkat Keyakinan: **{confidence:.2f}%**")

        st.subheader("Detail Probabilitas untuk Setiap Kelas:")
        # Tampilkan probabilitas untuk setiap kelas
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: **{prediction[0][i]*100:.2f}%**")

    except Exception as e:
        st.error("⚠️ **Terjadi kesalahan saat melakukan prediksi!**")
        st.write("Ini mungkin disebabkan oleh format gambar yang tidak terduga atau masalah pada model.")
        st.exception(e) # Tampilkan traceback error prediksi