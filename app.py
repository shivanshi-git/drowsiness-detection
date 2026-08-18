import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

from models.model_factory import get_model, count_parameters
from models.eye_model import EyeStateModel
from models.face_model import FaceDrowsinessModel
from models.fusion_engine import HierarchicalFusionEngine
from xai.grad_cam import GradCAM
from xai.visualizer import overlay_heatmap
from xai.hierarchical_xai import HierarchicalXAIVisualizer
from utils.face_mesh import FaceMeshAnalyzer

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

model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    options=['hierarchical_fusion', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'mobilenet_v2', 'mobilenet_v3', 'efficientnet_b0', 'custom_cnn', 'vit_tiny', 'yolov5'],
    index=0
)

confidence_threshold = st.sidebar.slider("Drowsiness Alert Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
cam_alpha = st.sidebar.slider("Grad-CAM Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Initialize Model & MediaPipe Analyzer
@st.cache_resource
def load_selected_model(name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'hierarchical_fusion':
        eye_m = EyeStateModel(pretrained=True).to(device)
        eye_m.eval()
        face_m = FaceDrowsinessModel(pretrained=True).to(device)
        face_m.eval()
        st.sidebar.success("✓ Loaded Hierarchical System (Eye State + Face Model)")
        return (eye_m, face_m), face_m.target_layer, device
    elif name == 'yolov5':
        custom_path = os.path.join("yolov5", "runs", "train", "exp15", "weights", "last.pt")
        if os.path.exists(custom_path):
            yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_path, force_reload=False)
            st.sidebar.success("✓ Loaded custom trained YOLOv5 weights")
        else:
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            st.sidebar.info("ℹ Using pretrained YOLOv5s model")
        yolo_model = yolo_model.to(device)
        yolo_model.eval()
        return yolo_model, None, device

    model, target_layer = get_model(model_name=name, num_classes=2, pretrained=True)
    checkpoint_path = os.path.join("checkpoints", name, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"✓ Loaded trained weights from checkpoint ({name})")
    else:
        st.sidebar.info(f"ℹ Using pre-trained ImageNet backbone ({name})")
    model = model.to(device)
    model.eval()
    return model, target_layer, device

model, target_layer, device = load_selected_model(model_choice)
face_analyzer = FaceMeshAnalyzer()

# Sidebar Model Metrics
if model_choice == 'hierarchical_fusion':
    params_m = 23.4
elif model_choice == 'yolov5':
    params_m = 7.07
else:
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

        if model_choice == 'hierarchical_fusion':
            eye_m, face_m = model
            pil_img = Image.fromarray(img_rgb)
            transform_224 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor_224 = transform_224(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                p_eye_closed = eye_m.get_eye_closed_prob(tensor_224).item()
                p_face_probs = face_m.get_face_probs(tensor_224).squeeze(0)
                p_face_drowsy = p_face_probs[1].item()
                p_yawn = p_face_probs[2].item() if p_face_probs.shape[0] > 2 else 0.0

            ear, mar, is_closed, is_yawning, annotated_bgr = face_analyzer.process_frame(img_bgr)
            fusion_engine = HierarchicalFusionEngine()
            score, is_drowsy, evidence = fusion_engine.evaluate(p_eye_closed, p_face_drowsy, p_yawn, ear, mar)

            h_xai = HierarchicalXAIVisualizer(face_m)
            blended_bgr, color_heatmap = h_xai.generate_evidence_overlay(img_bgr, tensor_224, alpha=cam_alpha)
            blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
            color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(color_heatmap_rgb, caption="Multi-Region Activation Heatmap", use_container_width=True)
            with col3:
                st.image(blended_rgb, caption="Defensible Evidence Overlay (Eye + Mouth Focus)", use_container_width=True)

            st.markdown("---")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                if is_drowsy:
                    st.markdown('<div class="alert-drowsy">🚨 DROWSY DRIVER DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="alert-normal">✅ DRIVER ALERT / NORMAL</div>', unsafe_allow_html=True)
            with mcol2:
                st.metric("Fusion Drowsiness Score", f"{score*100:.1f}%")
            with mcol3:
                st.metric("Eye State Signal", evidence["eye_state"], delta=f"p_closed={p_eye_closed:.2f}")
            with mcol4:
                st.metric("Facial Expression", evidence["face_expression"], delta=f"MAR={mar:.2f}")

        elif model_choice == 'yolov5':
            yolo_results = model(img_bgr)
            blended_rgb = cv2.cvtColor(np.squeeze(yolo_results.render()), cv2.COLOR_BGR2RGB)
            with col2:
                st.image(blended_rgb, caption="YOLOv5 Object Detection Bounding Boxes", use_container_width=True)
            with col3:
                st.image(blended_rgb, caption="Processed Output Frame", use_container_width=True)
            st.markdown("---")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown('<div class="alert-normal">✅ YOLOv5 OBJECT DETECTION COMPLETED</div>', unsafe_allow_html=True)
            with mcol2:
                st.metric("Model Architecture", "YOLOv5s")
        else:
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
        while run_cam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam feed not accessible.")
                break

            if model_choice == 'yolov5':
                results = model(frame)
                annotated_bgr = np.squeeze(results.render())
                ear, mar, is_closed, is_yawning, _ = face_analyzer.process_frame(frame)
            elif model_choice == 'hierarchical_fusion':
                ear, mar, is_closed, is_yawning, annotated_bgr = face_analyzer.process_frame(frame)
            else:
                ear, mar, is_closed, is_yawning, annotated_bgr = face_analyzer.process_frame(frame)

            status_text = "DROWSY ALARM!" if is_closed or is_yawning else "ALERT"
            color = (0, 0, 255) if status_text == "DROWSY ALARM!" else (0, 255, 0)
            
            cv2.putText(annotated_bgr, f"Status: {status_text} | EAR: {ear:.2f} | MAR: {mar:.2f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cam_placeholder.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

with tab3:
    st.subheader("Model Benchmarking & Comparison Matrix")
    st.markdown("""
    | Model Architecture | Paradigm | Parameters | Expected FPS (CPU) | Grad-CAM Target Layer | Recommended Use Case |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Hierarchical Fusion** | 2-Stage Multi-Task Fusion | ~23.4M | 35 FPS | `layer4[-1]` (Dual Region) | Scientifically Defensible Production Drowsiness System |
    | **VGG16** | Classic Deep CNN | ~15.0M | 15 FPS | `features[28]` | Maximum Explainability & Feature Detail |
    | **ResNet18** | Residual ConvNet | ~11.7M | 40 FPS | `layer4[-1]` | High Accuracy & Balanced Inference |
    | **MobileNetV2** | Depthwise Mobile | ~3.5M | 55+ FPS | `features[-1]` | Real-Time Embedded Driver Hardware |
    | **EfficientNet-B0** | Compound NAS | ~5.3M | 35 FPS | `features[-1]` | SOTA Parameter-to-Accuracy Efficiency |
    | **Custom CNN** | Baseline 5-Layer | ~0.8M | 60+ FPS | `features[12]` | Fast Baseline Training from Scratch |
    | **ViT-Tiny** | Vision Transformer | ~5.7M | 25 FPS | `encoder.layers[-1]` | Global Attention & Facial Feature Interaction |
    | **YOLOv5s** | One-Stage Detector | ~7.1M | 45 FPS | N/A (Bounding Boxes) | Real-Time Object Detection & Multi-Driver Tracking |
    """)


