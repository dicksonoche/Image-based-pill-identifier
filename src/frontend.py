import streamlit as st
import requests
from PIL import Image
import io

st.title("Pill Identifier MVP")

# Upload or Webcam
option = st.radio("Input Method", ("Upload Image", "Use Webcam"))
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose a pill image...", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files).json()
else:
    cam_image = st.camera_input("Take a photo")
    if cam_image:
        image = Image.open(cam_image)
        st.image(image, caption="Captured Image")
        files = {"file": (cam_image.name, cam_image.getvalue(), "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files).json()

if 'response' in locals():
    st.write(f"Predicted Pill: {response['prediction']}")
    st.write(f"Confidence: {response['confidence']:.2f}%")
    st.write(response['message'])
    if 'overlay_image' in response:
        overlay = Image.open(io.BytesIO(response['overlay_image']))
        st.image(overlay, caption="Overlay for Review")