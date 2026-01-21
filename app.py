import streamlit as st
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

st.set_page_config(page_title="Mon IA Magique", layout="centered")

st.title("🎨 Mon Atelier d'Images Rapide")

@st.cache_resource
def charger_modele():
    # On utilise un modèle "Fast" qui demande beaucoup moins de mémoire
    model_id = "Lykon/dreamshaper-8-lcm" 
    
    # On charge le tuyau magique
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,
        safety_checker=None # On retire le capteur pour gagner de la place
    )
    return pipe

# On lance le chargement
try:
    gen_pipe = charger_modele()
except Exception as e:
    st.error(f"Le serveur est un peu fatigué, réessaie dans 1 minute. Erreur : {e}")

photo_entree = st.file_uploader("Choisis une photo", type=['png', 'jpg', 'jpeg'])
prompt_texte = st.text_input("Ton sortilège (en anglais)", "Cyberpunk style, neon lights, high quality")

if st.button("Lancer la Magie !"):
    if photo_entree and prompt_texte:
        image = Image.open(photo_entree).convert("RGB").resize((512, 512))
        
        with st.spinner("L'IA dessine..."):
            # Pour ce modèle rapide, on utilise peu d'étapes (steps)
            # Ça évite de faire chauffer le serveur gratuit
            resultat = gen_pipe(
                prompt=prompt_texte, 
                image=image, 
                strength=0.6, 
                num_inference_steps=4, # Très rapide !
                guidance_scale=1.0
            ).images[0]
            
            st.image(resultat, caption="Résultat Magique")
    else:
        st.warning("Il manque la photo ou le texte !")
