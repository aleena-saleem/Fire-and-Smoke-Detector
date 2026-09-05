import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="Fire & Smoke Detector",
    page_icon="🔥",
    layout="wide"
)

MODEL_PATH = "fire_smoke_best_model.pt"
CONFIDENCE = 0.25


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

st.markdown("""
<style>
[data-testid="stHeader"] {
background: #f2f1ee;
}
.stApp {
background: #f2f1ee;
}
.block-container {
padding-top: 2.5rem;
padding-bottom: 3rem;
max-width: 1400px;
}
.eyebrow {
color: #d9642f;
font-size: 13px;
font-weight: 700;
letter-spacing: 2px;
margin-bottom: 14px;
}
.hero-title {
font-size: 52px;
font-weight: 800;
line-height: 1.08;
letter-spacing: -1px;
color: #23201d;
margin-bottom: 18px;
}
.hero-subtitle {
color: #6b655f;
font-size: 16px;
line-height: 1.7;
max-width: 640px;
margin-bottom: 22px;
}
.status-pill {
display: inline-flex;
align-items: center;
gap: 8px;
padding: 8px 16px;
border-radius: 999px;
background: #fdece2;
border: 1px solid #f3c9a8;
color: #d9642f;
font-weight: 600;
font-size: 13px;
margin-bottom: 36px;
}
.dot {
width: 7px;
height: 7px;
border-radius: 50%;
background: #d9642f;
display: inline-block;
}
.overview-box {
background: #e7e5e1;
border: 1px solid #d8d5cf;
border-radius: 16px;
padding: 22px 24px;
margin-bottom: 22px;
}
.side-block {
padding-bottom: 18px;
margin-bottom: 18px;
border-bottom: 1px solid #e0dcd5;
}
.side-block:last-child {
border-bottom: none;
}
.side-heading {
font-size: 15px;
font-weight: 700;
color: #23201d;
margin-bottom: 6px;
}
.side-text {
color: #837c74;
font-size: 13.5px;
line-height: 1.6;
}
.panel-title {
font-size: 22px;
font-weight: 700;
color: #23201d;
margin-bottom: 4px;
}
.panel-desc {
color: #837c74;
font-size: 14px;
margin-bottom: 24px;
}
[data-testid="stFileUploaderDropzone"] {
border: 1px dashed #f0b088;
background: #ece9e4;
border-radius: 14px;
}
.result-item {
background: #e7e5e1;
border: 1px solid #d8d5cf;
padding: 14px 18px;
border-radius: 12px;
margin-bottom: 8px;
font-size: 14.5px;
color: #23201d;
}
.fire-label {
color: #d9642f;
font-weight: 700;
}
.footer {
text-align: center;
color: #a39c93;
font-size: 13px;
margin-top: 40px;
padding-top: 18px;
border-top: 1px solid #e0dcd5;
}
</style>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2.3], gap="large")

with left_col:

    st.markdown("""
    <div class="overview-box">
    <div class="side-heading">Project Overview</div>
    <div class="side-text">An AI-based vision system for detecting fire and smoke hazards in forested and outdoor scenes.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="side-block">
    <div class="side-heading">YOLOv8 Detection</div>
    <div class="side-text">A convolutional object detector trained to localize fire and smoke regions with bounding boxes.</div>
    </div>
    <div class="side-block">
    <div class="side-heading">Confidence Scoring</div>
    <div class="side-text">Every detection is returned with a confidence score reflecting the model's certainty.</div>
    </div>
    <div class="side-block">
    <div class="side-heading">Single-Pass Inference</div>
    <div class="side-text">The uploaded image is processed once through the model to identify all fire and smoke instances present.</div>
    </div>
    <div class="side-block">
    <div class="side-heading">Any Image Input</div>
    <div class="side-text">Works on any uploaded JPG or PNG — aerial, ground-level, or handheld camera shots.</div>
    </div>
    """, unsafe_allow_html=True)

with right_col:

    st.markdown("""
    <div class="eyebrow">COMPUTER VISION · HAZARD DETECTION</div>
    <div class="hero-title">Fire &<br>Smoke Detection</div>
    <div class="hero-subtitle">Upload an image and the model will locate visible fire and smoke regions using a trained YOLOv8 detector.</div>
    <div class="status-pill"><span class="dot"></span>Model Ready</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-title">Detect Fire & Smoke</div>
    <div class="panel-desc">Upload an image to run detection.</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        image_bgr = np.array(image)[:, :, ::-1]

        with st.spinner("Analyzing image..."):
            results = model.predict(
                source=image_bgr,
                conf=CONFIDENCE,
                imgsz=640,
                verbose=False
            )

        result = results[0]
        annotated_img = result.plot()
        annotated_img = annotated_img[:, :, ::-1]

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.image(annotated_img, caption="Detected Fire / Smoke", use_container_width=True)

        boxes = result.boxes

        st.markdown("<br>", unsafe_allow_html=True)

        if boxes is not None and len(boxes) > 0:
            st.success(f"{len(boxes)} object(s) detected.")

            for i, box in enumerate(boxes, start=1):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                st.markdown(
                    f'<div class="result-item"><b>Detection {i}</b> &nbsp; '
                    f'<span class="fire-label">{label.title()}</span> &nbsp; • &nbsp; '
                    f'Confidence: <b>{conf * 100:.1f}%</b></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No fire or smoke detected in this image.")

st.markdown("""
<div class="footer">Fire & Smoke Detection System &nbsp; • &nbsp; YOLOv8 Computer Vision</div>
""", unsafe_allow_html=True)