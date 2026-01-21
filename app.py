import streamlit as st
import requests
import base64
import time
from io import BytesIO
from PIL import Image

# Configuration de la page pour mobile
st.set_page_config(page_title="IA Magique", layout="centered")

st.title("🎨 Mon Studio Photo IA")

# 1. Clé et Adresse du "Géant"
HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Les fonctions magiques (Le savoir-faire de l'expert)
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def interroger_ia(prompt, image_base64):
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": image_base64,
            "strength": 0.5,
            "num_inference_steps": 30
        }
    }
    
    # On envoie la demande
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # Si le géant dort (Erreur 503), on attend qu'il se réveille
    if response.status_code == 503:
        with st.status("Le géant se réveille... patiente un instant.", expanded=True):
            # On demande au géant combien de temps il lui faut
            attente = response.json().get('estimated_time', 20)
            time.sleep(attente)
            # On réessaie
            response = requests.post(API_URL, headers=headers, json=payload)
            
    return response.content

# 3. L'interface (Ce que tu vois sur ton Android)
photo = st.file_uploader("Prends ou choisis une photo", type=['jpg', 'jpeg', 'png'])
style = st.selectbox("Choisis un style", [
    "Cyberpunk, neon city background, high detail",
    "Anime style, studio ghibli, vibrant colors",
    "Ancient Greek statue, white marble",
    "Professional 3D render, Pixar style",
    "Epic fantasy, oil painting, masterpiece"
])

if st.button("🚀 Lancer la transformation"):
    if photo and HF_TOKEN:
        # On prépare l'image
        img_input = Image.open(photo).convert("RGB")
        img_input.thumbnail((768, 768))
        img_b64 = image_to_base64(img_input)
        
        with st.spinner("L'IA travaille sur ton image..."):
            result_bytes = interroger_ia(style, img_b64)
            
            try:
                # On essaie d'afficher le résultat
                img_output = Image.open(BytesIO(result_bytes))
                st.image(img_output, caption="Ta nouvelle image magique !")
                
                # Bouton de téléchargement pour ton téléphone
                buffered = BytesIO()
                img_output.save(buffered, format="PNG")
                st.download_button(
                    label="💾 Enregistrer l'image",
                    data=buffered.getvalue(),
                    file_name="mon_image_ia.png",
                    mime="image/png"
                )
            except:
                st.error("Le résultat n'est pas une image. Réessaie, le géant a eu un petit hoquet !")
    else:
        st.warning("Vérifie que tu as mis une photo et configuré ton Token !")
