import streamlit as st
import numpy as np
import pickle
from PIL import Image

st.set_page_config(
    page_title="Face Similarity AI",
    page_icon="🧠",
    layout="centered"
)

# Load PCA Model
with open("pca_model.pkl", "rb") as f:
    model = pickle.load(f)

pca = model["pca"]
img_size = model["img_size"]

st.title("👶➡️🧑 Face Similarity AI")
st.write("Bandingkan kemiripan wajah masa kecil dan dewasa menggunakan PCA")

foto1 = st.file_uploader(
    "Upload Foto Masa Kecil",
    type=["jpg", "jpeg", "png"]
)

foto2 = st.file_uploader(
    "Upload Foto Dewasa",
    type=["jpg", "jpeg", "png"]
)

if foto1:
    st.image(foto1, caption="Foto Masa Kecil", width=250)

if foto2:
    st.image(foto2, caption="Foto Dewasa", width=250)


def preprocess(uploaded_file):
    img = Image.open(uploaded_file)

    img = img.convert("L")
    img = img.resize(img_size)

    arr = np.array(img, dtype=np.float32)

    arr = arr / 255.0

    return arr.flatten().reshape(1, -1)


if st.button("Bandingkan"):

    if foto1 is None or foto2 is None:
        st.warning("Upload kedua foto terlebih dahulu.")

    else:

        with st.spinner("Menganalisis wajah..."):

            try:

                img1 = preprocess(foto1)
                img2 = preprocess(foto2)

                feat1 = pca.transform(img1)
                feat2 = pca.transform(img2)

                distance = np.linalg.norm(feat1 - feat2)

                similarity = max(
                    0,
                    min(
                        100,
                        100 - (distance * 10)
                    )
                )

                st.subheader("Hasil Analisis")

                st.metric(
                    "Tingkat Kemiripan",
                    f"{similarity:.2f}%"
                )

                st.progress(int(similarity))

                if similarity >= 80:
                    st.success("Sangat Mirip")
                elif similarity >= 60:
                    st.warning("Cukup Mirip")
                else:
                    st.error("Kurang Mirip")

            except Exception as e:
                st.error(f"Error: {e}")