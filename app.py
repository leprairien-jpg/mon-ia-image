import streamlit as st
import requests
import base64
import time
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Mon IA Perso", layout="centered")

# --- INTERFACE ---
st.title("🎨 Studio Créatif Libre")
st.write("Tu as le contrôle total ici !")

# --- CONFIGURATION ---
HF_TOKEN = st.secrets["HF_TOKEN"]
# Retour à une version plus stable et obéissante pour l'image-to-image
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def interroger_ia(prompt, image_base64, puissance):
    # On force l'IA à vraiment regarder l'image
    payload = {
        "inputs": prompt,
        "parameters": {
            "image": image_base64,
            "strength": puissance, # Contrôle la ressemblance
            "num_inference_steps": 30,
            "guidance_scale": 7.5 # Force l'obéissance au texte
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response

# --- ZONE DE RÉGLAGES ---
photo = st.file_uploader("1. Choisis ta photo", type=['jpg', 'jpeg', 'png'])

# ICI TU AS LA MAIN SUR LE TEXTE
prompt_libre = st.text_area("2. Ton sortilège (écris ce que tu veux en anglais)", 
                           "Convert this person into a heroic viking, cinematic lighting, highly detailed")

# ICI TU AS LA MAIN SUR LA RESSEMBLANCE
puissance_slider = st.slider("3. Force du changement (0.1 = presque pareil, 0.9 = totalement différent)", 0.1, 0.9, 0.5)

if st.button("🚀 Transformer"):
    if photo and prompt_libre:
        img_input = Image.open(photo).convert("RGB")
        img_input.thumbnail((512, 512)) # Taille parfaite pour ce modèle
        img_b64 = image_to_base64(img_input)
        
        with st.spinner("L'IA travaille..."):
            response = interroger_ia(prompt_libre, img_b64, puissance_slider)
            
            if response.status_code == 200:
                img_output = Image.open(BytesIO(response.content))
                st.image(img_output, caption="Résultat")
                
                # Option de téléchargement
                buf = BytesIO()
                img_output.save(buf, format="PNG")
                st.download_button("💾 Enregistrer", buf.getvalue(), "image.png", "image/png")
            
            elif response.status_code == 503:
                st.info("Le modèle finit de charger... Réessaie dans 10 secondes.")
            else:
                st.error(f"Erreur technique : {response.status_code}")
    else:
        st.warning("N'oublie pas la photo et le texte !")
