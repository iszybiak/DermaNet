import streamlit as st
import requests

st.title("🔬 AI Dermatolog – Rozpoznawanie chorób skóry")

uploaded_file = st.file_uploader("Wgraj zdjęcie skóry", type=["jpg", "png"])

if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post("http://127.0.0.1:5000/predict", files=files)
    st.write(f"**Diagnoza:** {response.json()['disease']}")