import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 1. Le Titre de notre Appli
st.title("🎨 Mon Atelier d'Images Magiques")
st.write("Mélange une photo et un texte pour créer du grand art !")

# 2. Configuration du Cerveau (On charge le modèle gratuit)
@st.cache_resource # Pour ne pas recharger le cerveau à chaque fois
def charger_modele():
    model_id = "runwayml/stable-diffusion-v1-5"
    # On utilise le processeur (CPU) car c'est gratuit sur les serveurs
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipe

gen_pipe = charger_modele()

# 3. La zone de téléchargement
photo_entree = st.file_uploader("Dépose ta photo ici", type=['png', 'jpg', 'jpeg'])
prompt_texte = st.text_input("Que doit-on ajouter ou changer ? (en anglais)", "A fantasy oil painting style")
puissance = st.slider("Force du changement (0 = pas de changement, 1 = tout nouveau)", 0.0, 1.0, 0.5)

if st.button("Lancer la Magie !"):
    if photo_entree is not None:
        # On prépare l'image
        image = Image.open(photo_entree).convert("RGB")
        image = image.resize((512, 512)) # On la taille pour que l'IA ne fatigue pas
        
        with st.spinner("Le robot réfléchit..."):
            # L'IA génère l'image
            resultat = gen_pipe(prompt=prompt_texte, image=image, strength=puissance, guidance_scale=7.5).images[0]
            
            # On montre le résultat
            st.image(resultat, caption="Tadam ! Voici ton image.")
    else:
        st.error("N'oublie pas de mettre une photo !")
