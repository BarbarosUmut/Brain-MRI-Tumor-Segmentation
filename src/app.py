
import streamlit as st
import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms

# Add src to path to allow imports
sys.path.append(os.path.abspath("src"))

from unet import UNet
from utils import calculate_tumor_area, calculate_tumor_location, overlay_mask
from ollama_client import OllamaClient

# Set page config
st.set_page_config(layout="wide", page_title="Brain Tumor Analysis AI")

@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning(f"Model file not found at {model_path}. Using untrained model for demonstration.")
    
    model.to(device)
    model.eval()
    return model, device

def process_image(image, device):
    # Eğitimde kullandığımız Mean ve Std değerlerinin aynısı buraya ekledik
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor

def main():
    st.title(" Brain Tumor Segmentation & Analysis")
    st.markdown("### AI-Powered Segmentation with U-Net and Ollama")

    # Sidebar
    st.sidebar.header("Configuration")
    model_path = st.sidebar.text_input("Model Path", "models/unet_best.pth")
    ollama_model = st.sidebar.text_input("Ollama Model", "llama3")
    
    # Load model
    model, device = load_model(model_path)
    
    # Initialize Ollama
    ollama_client = OllamaClient(model=ollama_model)

    # Main Interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("MRI Image Upload")
        uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["tif", "png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded MRI", use_column_width=True)
            
            # Predict
            if st.button("Segment Tumor"):
                with st.spinner("Segmenting..."):
                    input_tensor = process_image(image, device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.sigmoid(output).squeeze().cpu().numpy()
                    
                    # Store in session state
                    st.session_state['probs'] = probs
                    st.session_state['original_image'] = np.array(image.resize((256, 256)))
                    st.session_state['has_segmented'] = True

    with col2:
        st.subheader("Analysis Results")
        if 'has_segmented' in st.session_state and st.session_state['has_segmented']:
            probs = st.session_state['probs']
            original_image = st.session_state['original_image']
            
            # Display Mask
            mask_bin = (probs > 0.5).astype(np.uint8)
            
            # Overlay
            overlay = overlay_mask(original_image, probs, color=(255, 0, 0), alpha=0.4)
            st.image(overlay, caption="Tumor Segmentation (Overlay)", use_column_width=True)
            
            # Metrics
            area = calculate_tumor_area(probs)
            if area > 0:
                #Eğer tümör bulunuyorsa 
                overlay= overlay_mask(original_image, probs, color=(255, 0, 0), alpha=0.4)
                st.image(overlay, caption="Tumor Segmentation (Overlay)", use_column_width=True)
                
                # Konumu hesapla ve yazdır
                location = calculate_tumor_location(probs)
                st.success(f"✅ Tumor Detected!")
                st.info(f"**Estimated Tumor Area:** {area} pixels")
                st.info(f"**Centroid Location (X, Y):** {location}")
                
                # Bu değişkeni session_state'e kaydet ki AI raporu için kullanabilelim
                st.session_state['current_location'] = location
                st.session_state['current_area'] = area
            else:
                # TÜMÖR BULUNAMADIYSA
                st.image(original_image, caption="No Tumor Detected", use_column_width=True)
                st.warning("⚠️ No tumor detected in this MRI slice. The mask is empty.")
                st.session_state['current_location'] = None
                st.session_state['current_area'] = 0

            location = calculate_tumor_location(probs)
            
            st.info(f"**Estimated Tumor Area:** {area} pixels")
            st.info(f"**Centroid Location:** {location}")
            
            # AI Analysis
            st.divider()
            st.subheader(" AI Radiologist Assistant")
            if st.button("Generate Analysis Report"):
                with st.spinner(f"Consulting {ollama_model}..."):
                    report = ollama_client.analyze_tumor(area, location)
                    st.markdown("#### Report")
                    st.write(report)
                    
            # Reset
            if st.button("Clear Results"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
