import streamlit as st
import numpy as np
import pickle
import cv2

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

st.set_page_config(
    page_title="Deteksi Kemiripan Wajah",
    page_icon="🌷",
    layout="centered"
)

# ======================
# STYLE
# ======================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* BACKGROUND */
.stApp{
    background:linear-gradient(
        135deg,
        #312e81 0%,
        #5b21b6 50%,
        #7c3aed 100%
    );
    color:white;
}

/* GLOW EFFECT */
.glow1{
    position:fixed;

    top:-150px;
    left:-150px;

    width:500px;
    height:500px;

    background:#ff4ecd;

    border-radius:50%;

    filter:blur(120px);

    opacity:.4;

    z-index:0;
}

.glow2{
    position:fixed;

    bottom:-150px;
    right:-150px;

    width:500px;
    height:500px;

    background:#a855f7;

    border-radius:50%;

    filter:blur(120px);

    opacity:.4;

    z-index:0;
}

/* JUDUL */
.title{
    text-align:center;

    font-size:64px;
    font-weight:800;

    background:linear-gradient(
        90deg,
        #ffffff,
        #ffd6f8,
        #ff9de6
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    filter:drop-shadow(0 0 15px rgba(255,78,205,.6));
}

.subtitle{
    text-align:center;
    color:#e9d5ff;
    font-size:18px;
    margin-top:-10px;
    margin-bottom:30px;
}

/* CARD */
.glass{
    background:rgba(255,255,255,.10);

    backdrop-filter:blur(20px);

    border:1px solid rgba(255,255,255,.15);

    border-radius:24px;

    padding:24px;

    box-shadow:
    0 8px 30px rgba(0,0,0,.2),
    0 0 30px rgba(255,78,205,.15);
}

/* SCORE */
.score{
    font-size:110px;
    font-weight:800;
    text-align:center;

    background:linear-gradient(
        90deg,
        #ff8fd8,
        #ff4ecd,
        #c77dff
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    animation:pulseGlow 2.5s infinite;
}

@keyframes pulseGlow{

0%{
filter:
drop-shadow(0 0 10px #ff4ecd)
drop-shadow(0 0 25px #ff4ecd);
}

50%{
filter:
drop-shadow(0 0 20px #ff4ecd)
drop-shadow(0 0 50px #ff4ecd)
drop-shadow(0 0 80px #c77dff);
}

100%{
filter:
drop-shadow(0 0 10px #ff4ecd)
drop-shadow(0 0 25px #ff4ecd);
}

}

/* UPLOAD TITLE */
.upload-title{
    color:white;
    font-size:20px;
    font-weight:700;
    margin-bottom:10px;
}

/* AI INSIGHT */
.ai-card{
    background:rgba(255,255,255,.08);

    border:1px solid rgba(255,255,255,.15);

    backdrop-filter:blur(15px);

    border-radius:20px;

    padding:20px;

    margin-top:20px;
}
            
.ai-card h3{
    color:#ffd6f8;
    margin-bottom:10px;
}

.ai-card p{
    color:white;
    line-height:1.8;
}

.ai-title{
    color:#ffd6f8;
    font-size:24px;
    font-weight:700;
    margin-bottom:10px;
}

.ai-text{
    color:#ffffff;
    line-height:1.8;
    font-size:16px;
}

/* METRIC */
[data-testid="stMetricValue"]{
    color:white !important;
}

[data-testid="stMetricLabel"]{
    color:#f3e8ff !important;
}

/* BUTTON */
.stButton button{

    width:100%;

    border:none;

    border-radius:15px;

    color:white;

    font-weight:700;

    background:linear-gradient(
        90deg,
        #ff4ecd,
        #a855f7
    );

    box-shadow:
    0 0 20px rgba(255,78,205,.5);

    transition:.3s;
}

.stButton button:hover{

    transform:translateY(-3px);

    box-shadow:
    0 0 35px rgba(255,78,205,.8);

}

.block-container{
    position:relative;
    z-index:1;
}
            
.center{
    text-align:center;
    color:white;
    margin-top:25px;
}

img{
border-radius:20px;
}
            
            /* CENTER METRIC */

[data-testid="stMetric"]{
    text-align:center;
    background:rgba(255,255,255,.08);
    border:1px solid rgba(255,255,255,.15);
    border-radius:18px;
    padding:15px;
    backdrop-filter:blur(10px);
}

[data-testid="stMetricLabel"]{
    justify-content:center !important;
    color:#f3e8ff !important;
    font-weight:600;
}

[data-testid="stMetricValue"]{
    justify-content:center !important;
    color:white !important;
    font-weight:700;
}

/* HIDE STREAMLIT */
#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glow1"></div>
<div class="glow2"></div>
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
🌷 Deteksi Kemiripan Wajah
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

# PREVIEW FOTO

# PREVIEW FOTO

if foto1 or foto2:

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if foto1:

            st.markdown("""
            <div class="glass">
                <h4 style="text-align:center;color:white;">
                    👶 Foto Masa Kecil
                </h4>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                foto1,
                use_container_width=True
            )

    with col2:
        if foto2:

            st.markdown("""
            <div class="glass">
                <h4 style="text-align:center;color:white;">
                    🧑 Foto Dewasa
                </h4>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                foto2,
                use_container_width=True
            )

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
        
        st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

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