import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="IA Pro", layout="wide")
st.title("🎨 Transformation d'Image Haute Qualité")

# Configuration de l'accès
HF_TOKEN = st.secrets["HF_TOKEN"]
# Utilisation de SDXL (le modèle le plus puissant disponible gratuitement)
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def interroger_ia(prompt, image_base64):
    # On envoie le prompt ET l'image de référence
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": image_base64,
            "strength": 0.65, # 0.1 = proche de la photo, 0.9 = très créatif
            "num_inference_steps": 30
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

col1, col2 = st.columns(2)

with col1:
    photo = st.file_uploader("Ta photo source", type=['jpg', 'jpeg', 'png'])
    style = st.selectbox("Style souhaité", [
        "Cinematic, highly detailed, masterpiece",
        "Cyberpunk style, neon lights, 8k",
        "Oil painting, Van Gogh style, thick brushstrokes",
        "Disney Pixar 3D animation style",
        "Professional photography, portrait, soft lighting"
    ])
    prompt_perso = st.text_input("Ajoute un détail précis (ex: 'wearing sunglasses')", "")

if st.button("🚀 Transformer l'image"):
    if photo:
        img_input = Image.open(photo).convert("RGB")
        img_input.thumbnail((768, 768)) # Taille optimale pour SDXL
        img_b64 = image_to_base64(img_input)
        
        full_prompt = f"{style}, {prompt_perso}"
        
        with st.spinner("L'IA analyse et redessine..."):
            result_bytes = interroger_ia(full_prompt, img_b64)
            try:
                img_output = Image.open(BytesIO(result_bytes))
                with col2:
                    st.image(img_output, caption="Résultat Haute Qualité")
            except:
                st.error("L'IA est en train de chauffer (chargement du modèle). Réessaie dans 30 secondes !")
    else:
        st.warning("Ajoute une photo d'abord !")
