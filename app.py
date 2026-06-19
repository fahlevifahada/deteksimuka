import streamlit as st
from deepface import DeepFace
import tempfile

st.set_page_config(page_title="Face Similarity AI")

st.title("👶➡️🧑 Face Similarity AI")
st.write("Bandingkan kemiripan wajah masa kecil dan dewasa")

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

if st.button("Bandingkan"):

    if foto1 is None or foto2 is None:
        st.warning("Upload kedua foto terlebih dahulu.")
    else:

        with st.spinner("Menganalisis wajah..."):

            file1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            file1.write(foto1.read())
            file1.close()

            file2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            file2.write(foto2.read())
            file2.close()

            try:
                hasil = DeepFace.verify(
                    img1_path=file1.name,
                    img2_path=file2.name,
                    enforce_detection=False
                )

                distance = hasil["distance"]

                similarity = max(0, (1 - distance) * 100)

                st.subheader("Hasil Analisis")

                st.metric(
                    "Tingkat Kemiripan",
                    f"{similarity:.2f}%"
                )

                if similarity >= 80:
                    st.success("Sangat Mirip")
                elif similarity >= 60:
                    st.warning("Cukup Mirip")
                else:
                    st.error("Kurang Mirip")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")