import streamlit as st
import numpy as np
import pickle
import cv2

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

st.set_page_config(
    page_title="Neural Face Match",
    page_icon="🧬",
    layout="wide"
)

# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

.stApp{
background:
linear-gradient(
135deg,
#0f172a,
#111827,
#1e1b4b
);
color:white;
}

.main-title{
text-align:center;
font-size:50px;
font-weight:800;
color:#c084fc;
text-shadow:0 0 25px #9333ea;
}

.subtitle{
text-align:center;
color:#cbd5e1;
margin-bottom:30px;
}

.result-box{
padding:20px;
border-radius:20px;
background:rgba(255,255,255,0.05);
backdrop-filter:blur(10px);
text-align:center;
margin-top:20px;
}

.big-score{
font-size:70px;
font-weight:900;
color:#a855f7;
text-shadow:0 0 20px #9333ea;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================

with open("pca_model.pkl","rb") as f:
    model = pickle.load(f)

pca = model["pca"]
img_size = model["img_size"]

# ==========================
# FACE DETECTION
# ==========================

def preprocess(uploaded_file):

    img = Image.open(uploaded_file)

    rgb = np.array(img.convert("RGB"))

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) > 0:

        x,y,w,h = faces[0]

        gray = gray[y:y+h,x:x+w]

    gray = cv2.resize(
        gray,
        img_size
    )

    gray = gray / 255.0

    return gray.flatten().reshape(1,-1)

# ==========================
# HEADER
# ==========================

st.markdown(
    '<div class="main-title">🧬 Neural Face Match</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Bandingkan foto masa kecil dan dewasa menggunakan PCA Eigenfaces</div>',
    unsafe_allow_html=True
)

# ==========================
# UPLOAD
# ==========================

col1,col2 = st.columns(2)

with col1:
    foto1 = st.file_uploader(
        "📷 Foto Masa Kecil",
        type=["jpg","jpeg","png"]
    )

with col2:
    foto2 = st.file_uploader(
        "📷 Foto Dewasa",
        type=["jpg","jpeg","png"]
    )

if foto1:
    st.image(foto1,width=250)

if foto2:
    st.image(foto2,width=250)

# ==========================
# ANALYZE
# ==========================

if st.button("🚀 Analisis Wajah"):

    if foto1 is None or foto2 is None:

        st.warning(
            "Upload kedua foto terlebih dahulu"
        )

    else:

        progress = st.progress(0)

        for i in range(100):
            progress.progress(i+1)

        try:

            img1 = preprocess(foto1)
            img2 = preprocess(foto2)

            vec1 = pca.transform(img1)
            vec2 = pca.transform(img2)

            cos_sim = float(
                cosine_similarity(
                    vec1,
                    vec2
                )[0][0]
            )

            euc_dist = float(
                euclidean(
                    vec1[0],
                    vec2[0]
                )
            )

            cos_pct = max(
                0,
                min(
                    100,
                    ((cos_sim + 1)/2)*100
                )
            )

            euc_pct = max(
                0,
                min(
                    100,
                    (1-(euc_dist/60))*100
                )
            )

            similarity = (
                cos_pct +
                euc_pct
            ) / 2

            st.markdown(
                f"""
                <div class="result-box">

                <div class="big-score">
                {similarity:.1f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Cosine Similarity",
                f"{cos_sim:.4f}"
            )

            st.metric(
                "Euclidean Distance",
                f"{euc_dist:.4f}"
            )

            st.progress(
                int(similarity)
            )

            if similarity >= 80:

                st.success(
                    "✅ Sangat Mirip"
                )

            elif similarity >= 60:

                st.warning(
                    "🟡 Cukup Mirip"
                )

            else:

                st.error(
                    "❌ Kurang Mirip"
                )

        except Exception as e:

            st.error(
                f"Error : {e}"
            )