import streamlit as st
import requests
import base64
import time
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Studio Photo IA", layout="centered")

# --- CONFIGURATION ---
HF_TOKEN = st.secrets["HF_TOKEN"]
# Utilisation de la version 2.1, plus stable et sans erreur 410
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def interroger_ia(prompt, image_base64, puissance):
    # Format spécifique pour SD 2.1
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": image_base64,
            "strength": puissance,
            "num_inference_steps": 25,
            "guidance_scale": 9.0
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# --- INTERFACE ---
st.title("🎨 Mon Studio IA (Version Stable)")

photo = st.file_uploader("1. Ta photo", type=['jpg', 'jpeg', 'png'])
prompt_libre = st.text_area("2. Ton souhait (en anglais)", "A futuristic cyborg, high quality, cinematic")
puissance = st.slider("3. Ressemblance (0.1 = identique, 0.9 = méconnaissable)", 0.1, 0.9, 0.5)

if st.button("🚀 Transformer"):
    if photo and prompt_libre:
        if not HF_TOKEN:
            st.error("N'oublie pas d'ajouter ton HF_TOKEN dans les Secrets !")
            st.stop()
            
        img_input = Image.open(photo).convert("RGB")
        img_input.thumbnail((768, 768)) # SD 2.1 préfère cette taille
        img_b64 = image_to_base64(img_input)
        
        with st.spinner("L'IA travaille..."):
            response = interroger_ia(prompt_libre, img_b64, puissance)
            
            if response.status_code == 200:
                img_output = Image.open(BytesIO(response.content))
                st.image(img_output, use_column_width=True)
                
                # Sauvegarde
                buf = BytesIO()
                img_output.save(buf, format="PNG")
                st.download_button("💾 Enregistrer", buf.getvalue(), "image_ia.png")
                
            elif response.status_code == 503:
                st.warning("Le modèle charge ses batteries... Réessaie dans 20 secondes !")
            elif response.status_code == 401:
                st.error("Ta clé (Token) n'est pas bonne. Vérifie tes Secrets !")
            else:
                st.error(f"Erreur {response.status_code} : Hugging Face est un peu fatigué.")
    else:
        st.warning("Photo + Texte obligatoires !")
