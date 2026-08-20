import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

from models.model_factory import get_model, count_parameters
from xai.grad_cam import GradCAM
from xai.visualizer import overlay_heatmap
from utils.face_mesh import FaceMeshAnalyzer
from utils.temporal_buffer import TemporalPERCLOSBuffer

st.set_page_config(
    page_title="Driver Drowsiness Detection with XAI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .alert-drowsy {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .alert-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 Driver Drowsiness Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning Architecture Comparison & Explainable AI (Grad-CAM) Visualizer</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ System Configuration")

model_choice = 'resnet18'
st.sidebar.info("Canonical model: ResNet18")

confidence_threshold = st.sidebar.slider("Drowsiness Alert Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
cam_alpha = st.sidebar.slider("Grad-CAM Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Initialize Model & MediaPipe Analyzer
@st.cache_resource
def load_selected_model(name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, target_layer = get_model(model_name=name, num_classes=2, pretrained=False)
    checkpoint_candidates = [
        os.path.join("saved_models", f"{name}_best_model.pth"),
        os.path.join("saved_models", f"{name}_drowsiness_model.pth"),
    ]
    checkpoint_path = next((path for path in checkpoint_candidates if os.path.exists(path)), None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"✓ Loaded trained weights from checkpoint ({name})")
    else:
        st.sidebar.warning(f"⚠ No trained checkpoint found for {name}; using randomly initialized weights")
    model = model.to(device)
    model.eval()
    return model, target_layer, device

model, target_layer, device = load_selected_model(model_choice)
face_analyzer = FaceMeshAnalyzer()

# Sidebar Model Metrics
params_m = count_parameters(model)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Selected Model Specs")
st.sidebar.write(f"• **Architecture**: `{model_choice.upper()}`")
st.sidebar.write(f"• **Parameters**: `{params_m:.2f} Million`")
st.sidebar.write(f"• **Execution Device**: `{device.type.upper()}`")

# Main Interface Tabs
tab1, tab2, tab3 = st.tabs(["🖼 Single Image Analysis & XAI", "🎥 Real-Time Camera / Video Stream", "📈 Architecture Comparison Matrix"])

with tab1:
    st.subheader("Image Drowsiness Diagnostic & Grad-CAM Heatmap")
    uploaded_file = st.file_uploader("Upload a Driver Image (JPG, PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_rgb, caption="Input Driver Image", use_container_width=True)

        if model_choice == 'resnet18':
            # Preprocess for model
            pil_img = Image.fromarray(img_rgb)
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor_img = transform(pil_img).unsqueeze(0).to(device)

            # Compute Grad-CAM
            grad_cam = GradCAM(model, target_layer)
            heatmap, class_idx, confidence = grad_cam.generate_heatmap(tensor_img)
            grad_cam.remove_hooks()

            class_names = ["Alert", "Drowsy"]
            pred_label = class_names[class_idx]

            # Compute Geometric EAR & MAR
            ear, mar, is_closed, is_yawning, annotated_bgr = face_analyzer.process_frame(img_bgr)

            # Blend Heatmap
            blended_bgr, color_heatmap = overlay_heatmap(img_bgr, heatmap, alpha=cam_alpha)
            blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
            color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(color_heatmap_rgb, caption=f"Raw Grad-CAM Heatmap ({model_choice.upper()})", use_container_width=True)

            with col3:
                st.image(blended_rgb, caption="XAI Overlay (Model Focus)", use_container_width=True)

            st.markdown("---")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)

            is_drowsy = (pred_label == "Drowsy" and confidence >= confidence_threshold) or is_closed

            with mcol1:
                if is_drowsy:
                    st.markdown('<div class="alert-drowsy">🚨 DROWSY DRIVER DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-normal">✅ DRIVER ALERT / NORMAL</div>', unsafe_allow_html=True)

            with mcol2:
                st.metric("Model Confidence", f"{confidence*100:.1f}%")

            with mcol3:
                st.metric("Eye Aspect Ratio (EAR)", f"{ear:.3f}", delta="Closed" if is_closed else "Open")

            with mcol4:
                st.metric("Mouth Aspect Ratio (MAR)", f"{mar:.3f}", delta="Yawning" if is_yawning else "Normal")

with tab2:
    st.subheader("Live Camera / Video Stream Analysis")
    st.info("Click 'Start Camera Stream' to initiate real-time drowsiness detection with dynamic EAR tracking.")

    run_cam = st.checkbox("Start Camera Stream")
    cam_placeholder = st.empty()

    if run_cam:
        cap = cv2.VideoCapture(0)
        perclos_buffer = TemporalPERCLOSBuffer()
        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam feed not accessible.")
                break

            ear, mar, is_closed, is_yawning, annotated_bgr = face_analyzer.process_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
            with torch.inference_mode():
                model_probability = torch.softmax(model(frame_tensor), dim=1)[0, 1].item()

            temporal_result = perclos_buffer.update(
                model_probability >= confidence_threshold or is_closed or is_yawning
            )
            status_text = temporal_result["state"]
            color = {
                "ALERT": (0, 255, 0),
                "WARNING": (0, 165, 255),
                "DANGER": (0, 0, 255),
            }[status_text]
            
            cv2.putText(annotated_bgr, f"State: {status_text} | PERCLOS: {temporal_result['perclos']:.0%} | P(Drowsy): {model_probability:.2f} | EAR: {ear:.2f} | MAR: {mar:.2f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cam_placeholder.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

with tab3:
    st.subheader("Model Benchmarking & Comparison Matrix")
    st.markdown("""
    | Model Architecture | Paradigm | Parameters | Expected FPS (CPU) | Grad-CAM Target Layer | Recommended Use Case |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **ResNet18** | Residual ConvNet | ~11.7M | 40 FPS | `layer4[-1]` | High Accuracy & Balanced Inference |
    """)


