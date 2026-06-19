import streamlit as st
import numpy as np
import pickle
import cv2

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

st.set_page_config(
    page_title="Then & Now AI",
    page_icon="🌷",
    layout="centered"
)

# ======================
# STYLE
# ======================

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#fdf4ff,
#f5d0fe,
#ddd6fe
);
}

.title{
font-size:60px;
font-weight:900;
text-align:center;
background:linear-gradient(
90deg,
#ec4899,
#8b5cf6
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
text-align:center;
color:#6b7280;
margin-bottom:30px;
}

.glass{
background:rgba(255,255,255,.45);
backdrop-filter:blur(18px);
padding:25px;
border-radius:25px;
box-shadow:0 8px 30px rgba(0,0,0,.08);
}

.score{
font-size:90px;
font-weight:900;
text-align:center;
background:linear-gradient(
90deg,
#ec4899,
#8b5cf6
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.center{
text-align:center;
}

.ai-card{
background:white;
padding:20px;
border-radius:20px;
margin-top:20px;
border-left:6px solid #8b5cf6;
}

.stButton button{
width:100%;
height:60px;
border:none;
border-radius:20px;
font-size:18px;
font-weight:700;
background:linear-gradient(
90deg,
#ec4899,
#8b5cf6
);
color:white;
}

</style>
""", unsafe_allow_html=True)

# ======================
# MODEL
# ======================

with open("pca_model.pkl","rb") as f:
    model = pickle.load(f)

pca = model["pca"]
img_size = model["img_size"]

# ======================
# PREPROCESS
# ======================

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

# ======================
# HEADER
# ======================

st.markdown("""
<div class='title'>
🌷 Then & Now AI
</div>

<div class='subtitle'>
Bandingkan wajah masa kecil dan dewasa
menggunakan PCA Face Recognition
</div>
""", unsafe_allow_html=True)

# ======================
# UPLOAD
# ======================

st.markdown("<div class='glass'>", unsafe_allow_html=True)

foto1 = st.file_uploader(
    "👶 Upload Foto Masa Kecil",
    type=["jpg","jpeg","png"]
)

foto2 = st.file_uploader(
    "🧑 Upload Foto Dewasa",
    type=["jpg","jpeg","png"]
)

st.markdown("</div>", unsafe_allow_html=True)

# ======================
# ANALISIS
# ======================

if st.button("🚀 Analisis Kemiripan"):

    if foto1 is None or foto2 is None:

        st.warning(
            "Upload kedua foto terlebih dahulu."
        )

    else:

        with st.spinner(
            "AI sedang menganalisis wajah..."
        ):

            img1 = preprocess(foto1)
            img2 = preprocess(foto2)

            v1 = pca.transform(img1)
            v2 = pca.transform(img2)

            cos_sim = float(
                cosine_similarity(v1,v2)[0][0]
            )

            euc_dist = float(
                euclidean(v1[0],v2[0])
            )

            cos_pct = ((cos_sim + 1)/2)*100

            euc_pct = max(
                0,
                (1-(euc_dist/60))*100
            )

            similarity = (
                cos_pct * 0.7 +
                euc_pct * 0.3
            )

            similarity = max(
                0,
                min(
                    100,
                    similarity
                )
            )

        st.markdown(f"""
        <div class='glass'>

        <div class='score'>
        {similarity:.1f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(
            int(similarity)
        )

        if similarity >= 80:

            verdict = "✅ Sangat Mirip"

            insight = """
            Struktur wajah menunjukkan
            kemiripan yang sangat tinggi.
            Perubahan usia tidak terlalu
            mempengaruhi pola utama wajah.
            """

        elif similarity >= 60:

            verdict = "🟡 Cukup Mirip"

            insight = """
            Terdapat kemiripan yang cukup kuat,
            namun perubahan usia dan ekspresi
            mempengaruhi hasil analisis.
            """

        else:

            verdict = "❌ Kurang Mirip"

            insight = """
            Pola wajah yang terdeteksi cukup
            berbeda sehingga sistem menemukan
            kemiripan yang rendah.
            """

        st.markdown(
            f"<h2 class='center'>{verdict}</h2>",
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class='ai-card'>

        <h3>🤖 AI Insight</h3>

        <p>{insight}</p>

        </div>
        """, unsafe_allow_html=True)

        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric(
                "Cosine",
                f"{cos_sim:.3f}"
            )

        with col2:
            st.metric(
                "Distance",
                f"{euc_dist:.2f}"
            )

        with col3:
            st.metric(
                "Confidence",
                f"{similarity:.1f}%"
            )