import streamlit as st
import requests
import base64
import time
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="IA Photo Pro", layout="centered")

# --- CONFIGURATION ---
HF_TOKEN = st.secrets["HF_TOKEN"]
# Ce modèle est ultra-fiable et excellent pour les visages/photos
API_URL = "https://api-inference.huggingface.co/models/SG161222/Realistic_Vision_V5.1_noVAE"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def interroger_ia(prompt, image_base64, puissance):
    # On ajoute "options": {"wait_for_model": True} pour éviter les erreurs de chargement
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": image_base64,
            "strength": puissance,
            "num_inference_steps": 30,
        },
        "options": {"wait_for_model": True}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# --- INTERFACE ---
st.title("📸 Studio Photo IA (Mode Expert)")
st.write("Ce moteur est plus puissant et plus stable.")

photo = st.file_uploader("1. Dépose ta photo", type=['jpg', 'jpeg', 'png'])
prompt_libre = st.text_area("2. Description du changement (en anglais)", "a rugged viking warrior, cinematic lighting, high detail, 8k")
puissance = st.slider("3. Intensité de la transformation", 0.1, 0.9, 0.5)

if st.button("🚀 Transformer l'image"):
    if photo and prompt_libre:
        img_input = Image.open(photo).convert("RGB")
        img_input.thumbnail((512, 512)) 
        img_b64 = image_to_base64(img_input)
        
        with st.spinner("Le moteur de rendu démarre... (cela peut prendre 30s)"):
            response = interroger_ia(prompt_libre, img_b64, puissance)
            
            if response.status_code == 200:
                try:
                    img_output = Image.open(BytesIO(response.content))
                    st.image(img_output, use_container_width=True)
                    
                    buf = BytesIO()
                    img_output.save(buf, format="PNG")
                    st.download_button("💾 Sauvegarder", buf.getvalue(), "photo_ia.png")
                except:
                    st.error("L'image est arrivée corrompue. Réessaie une fois !")
            
            elif response.status_code == 410 or response.status_code == 404:
                st.error("Ce moteur est momentanément indisponible. Je cherche une autre route...")
            elif response.status_code == 503:
                st.info("Le moteur préchauffe... Presse le bouton à nouveau dans 10 secondes.")
            else:
                st.error(f"Erreur {response.status_code}. Vérifie ton Token Hugging Face.")
    else:
        st.warning("Photo et texte requis.")
