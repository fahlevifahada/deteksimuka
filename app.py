import streamlit as st
import numpy as np
import cv2
from sklearn.decomposition import PCA
from PIL import Image

st.set_page_config(page_title="Face PCA Similarity")

st.title("👶➡️🧑 Face Similarity (PCA Version)")
st.write("Bandingkan wajah menggunakan PCA (Principal Component Analysis)")

foto1 = st.file_uploader("Upload Foto Masa Kecil", type=["jpg", "jpeg", "png"])
foto2 = st.file_uploader("Upload Foto Dewasa", type=["jpg", "jpeg", "png"])

def preprocess(img_file):
    image = Image.open(img_file).convert("L")  # grayscale
    image = image.resize((100, 100))  # kecilkan biar ringan
    return np.array(image).flatten()

if foto1 and foto2:
    st.image([foto1, foto2], caption=["Kecil", "Dewasa"], width=200)

if st.button("Bandingkan"):

    if foto1 is None or foto2 is None:
        st.warning("Upload kedua foto dulu!")

    else:
        try:
            img1 = preprocess(foto1)
            img2 = preprocess(foto2)

            data = np.array([img1, img2])

            # PCA reduce ke 1 dimensi
            pca = PCA(n_components=1)
            reduced = pca.fit_transform(data)

            diff = abs(reduced[0] - reduced[1])

            similarity = max(0, (1 - diff) * 100)

            st.subheader("Hasil PCA")

            st.metric("Kemiripan", f"{similarity:.2f}%")

            if similarity >= 80:
                st.success("Sangat Mirip")
            elif similarity >= 60:
                st.warning("Cukup Mirip")
            else:
                st.error("Kurang Mirip")

        except Exception as e:
            st.error(f"Error: {e}")