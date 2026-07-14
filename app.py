import streamlit as st
import time
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Neura Vision AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*{
font-family:'Inter',sans-serif;
}

/* Hide Streamlit Branding */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}



/* Background */

.stApp{

background:
radial-gradient(circle at top left,#312E81 0%,transparent 30%),
radial-gradient(circle at top right,#1D4ED8 0%,transparent 35%),
radial-gradient(circle at bottom,#020617 10%,#020617 70%);

color:white;

}

/* Main Container */

.block-container{

padding-top:2rem;

padding-left:3rem;

padding-right:3rem;

}

/* Hero */

.hero{

background:rgba(255,255,255,.06);

border:1px solid rgba(255,255,255,.08);

border-radius:28px;

padding:45px;

backdrop-filter:blur(18px);

box-shadow:

0 0 45px rgba(99,102,241,.18);

margin-bottom:35px;

}

/* Heading */

.hero-title{

font-size:62px;

font-weight:800;

background:

linear-gradient(

90deg,

#60A5FA,

#8B5CF6,

#EC4899

);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

margin-bottom:12px;

}

/* Subtitle */

.hero-sub{

font-size:22px;

color:#CBD5E1;

font-weight:400;

}

/* Glass Cards */

.glass{

background:rgba(255,255,255,.05);

border-radius:22px;

padding:25px;

border:1px solid rgba(255,255,255,.08);

backdrop-filter:blur(18px);

box-shadow:

0 10px 30px rgba(0,0,0,.35);

}

/* Buttons */

.stButton>button{

width:100%;

height:58px;

border:none;

border-radius:15px;

font-size:18px;

font-weight:700;

color:white;

background:

linear-gradient(

90deg,

#6366F1,

#8B5CF6

);

transition:.3s;

}

.stButton>button:hover{

transform:translateY(-3px);

box-shadow:

0 0 35px rgba(99,102,241,.55);

}

/* Metrics */

[data-testid="metric-container"]{

background:rgba(255,255,255,.05);

border:1px solid rgba(255,255,255,.08);

padding:18px;

border-radius:18px;

}

/* Success */

.stSuccess{

border-radius:15px;

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#050816;

}

/* Footer */

.footer{

margin-top:60px;

text-align:center;

color:#94A3B8;

}

</style>

""", unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================

st.markdown("""

<div class="hero">

<div class="hero-title">

🧠 NEURA VISION AI

</div>

<div class="hero-sub">

Intelligent Image Caption Generator powered by
Vision Transformer + GPT-2

</div>

</div>

""", unsafe_allow_html=True)

# ============================================================
# DASHBOARD STATUS
# ============================================================

a,b,c,d = st.columns(4)

with a:
    st.metric("Status","🟢 ONLINE")

with b:
    st.metric("Encoder","ViT")

with c:
    st.metric("Decoder","GPT-2")

with d:
    st.metric("Framework","PyTorch")
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 NeuraVision AI")

    st.markdown("---")

    st.subheader("📌 About")

    st.write("""
Generate intelligent image captions using
Vision Transformer and GPT-2.
""")

    st.markdown("---")

    st.subheader("⚙ AI Engine")

    st.success("Encoder : Vision Transformer")

    st.success("Decoder : GPT-2")

    st.success("Framework : PyTorch")

    st.success("Status : Online")

    st.markdown("---")

    st.subheader("👩‍💻 Developer")

    st.write("Monica,Mounisha, Namitha ")

    st.write("BE Computer Science")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():

    model = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning",
        local_files_only=False
    )

    processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning",
        local_files_only=False
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning",
        local_files_only=False
    )

    model.eval()

    return model, processor, tokenizer


model, processor, tokenizer = load_model()

# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([1.05,0.95], gap="large")

# ============================================================
# LEFT PANEL
# ============================================================

with left:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

    image = None

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        width,height = image.size

        c1,c2 = st.columns(2)

        c1.metric("Width",f"{width}px")

        c2.metric("Height",f"{height}px")

        st.metric(
            "Size",
            f"{round(uploaded_file.size/1024,2)} KB"
        )

    st.markdown("</div>", unsafe_allow_html=True)
    # ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("🧠 AI Caption Generator")

    if uploaded_file:

        st.write("Ready to analyze your image.")

        generate = st.button("✨ Generate Caption")

        if generate:

            progress = st.progress(0)

            status = st.empty()

            # -----------------------------
            # Fake AI Pipeline Animation
            # -----------------------------

            status.info("🧠 Initializing Vision Transformer...")
            progress.progress(15)
            time.sleep(0.4)

            status.info("🖼 Extracting Image Features...")
            progress.progress(35)
            time.sleep(0.4)

            pixel_values = processor(
                images=image,
                return_tensors="pt"
            ).pixel_values

            status.info("⚡ Running Attention Layers...")
            progress.progress(60)
            time.sleep(0.4)

            start = time.time()

            output_ids = model.generate(
                pixel_values,
                max_length=30,
                num_beams=4
            )

            status.info("🤖 GPT-2 Language Decoder...")
            progress.progress(85)
            time.sleep(0.4)

            caption = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            end = time.time()

            progress.progress(100)

            status.success("✅ Caption Generated Successfully!")

            st.markdown("---")

            st.subheader("📝 Generated Caption")

            placeholder = st.empty()

            typed = ""

            for letter in caption:

                typed += letter

                placeholder.markdown(
                    f"""
<div class='glass' style='font-size:22px;font-weight:600;'>

{typed}▋

</div>
""",
                    unsafe_allow_html=True,
                )

                time.sleep(0.02)

            placeholder.markdown(
                f"""
<div class='glass' style='font-size:22px;font-weight:600;'>

{caption}

</div>
""",
                unsafe_allow_html=True,
            )

            st.markdown("")

            c1, c2 = st.columns(2)

            c1.metric(
                "⚡ Inference Time",
                f"{end-start:.2f} sec"
            )

            c2.metric(
                "Caption Length",
                f"{len(caption.split())} words"
            )

            st.download_button(
                "📥 Download Caption",
                data=caption,
                file_name="caption.txt",
                mime="text/plain"
            )

    else:

        st.info("👈 Upload an image to begin.")

    st.markdown("</div>", unsafe_allow_html=True)
    # ============================================================
# AI CONSOLE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🖥 AI Processing Console", expanded=False):

    console = """
[ SYSTEM ] Boot sequence complete...

[ INFO ] Vision Transformer initialized

[ INFO ] Image converted to RGB

[ INFO ] Splitting image into patches

[ INFO ] Computing self-attention

[ INFO ] Extracting visual embeddings

[ INFO ] Passing embeddings to GPT-2

[ INFO ] Generating caption...

[ SUCCESS ] Caption generated successfully
"""

    st.code(console, language="text")

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("## 🤖 AI Model Architecture")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
<div class="glass">

<h3>🧠 Encoder</h3>

<h2>Vision Transformer (ViT)</h2>

<p>

✔ Image Patch Embedding

<br>

✔ Multi-Head Self Attention

<br>

✔ Feature Extraction

</p>

</div>
""", unsafe_allow_html=True)

with col2:

    st.markdown("""
<div class="glass">

<h3>🤖 Decoder</h3>

<h2>GPT-2</h2>

<p>

✔ Language Modeling

<br>

✔ Caption Generation

<br>

✔ Natural Language Output

</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("## ⚙ How It Works")

step1, step2, step3, step4, step5 = st.columns(5)

step1.success("📤 Upload")

step2.info("🖼 Encode")

step3.info("🧠 Attention")

step4.info("🤖 GPT-2")

step5.success("📝 Caption")

# ============================================================
# FEATURE CARDS
# ============================================================

st.markdown("## 🚀 Features")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
<div class="glass">

### 📸 Image Upload

Supports JPG, JPEG and PNG.

</div>
""", unsafe_allow_html=True)

with f2:
    st.markdown("""
<div class="glass">

### ⚡ AI Captioning

Powered by Vision Transformer + GPT-2.

</div>
""", unsafe_allow_html=True)

with f3:
    st.markdown("""
<div class="glass">

### 📥 Export

Download generated captions instantly.

</div>
""", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div class="footer">

<h2>✨ NeuraVision AI</h2>

<p>

Intelligent Image Caption Generator

</p>

<p>

Built with ❤️ using

Python • Streamlit • PyTorch • Hugging Face

</p>

<br>

<p>

Developed by <b>Monica , Mounisha, Namitha </b>

</p>

</div>
""", unsafe_allow_html=True)
    