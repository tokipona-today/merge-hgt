import os
import streamlit as st
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import tempfile

st.set_page_config(
    page_title="Fusion de DEM",
    page_icon="🏔️",
    layout="wide"
)

st.title("Fusion de fichiers DEM (.hgt)")
st.write("Cette application permet de fusionner plusieurs fichiers DEM (Digital Elevation Model) en un seul fichier.")


def fusion_dem(fichiers_entree, fichier_sortie, methode_reechantillonnage="bilinear"):
    try:
        # Vérifier que tous les fichiers existent
        for fichier in fichiers_entree:
            if not os.path.exists(fichier):
                st.error(f"Erreur: Le fichier {fichier} n'existe pas.")
                return False

        # Ouvrir les fichiers raster
        src_files_to_mosaic = []
        for fp in fichiers_entree:
            try:
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)
                st.write(f"Fichier ouvert: {fp}")
            except Exception as e:
                st.error(f"Erreur lors de l'ouverture de {fp}: {str(e)}")
                # Fermer les fichiers déjà ouverts
                for src in src_files_to_mosaic:
                    src.close()
                return False

        if not src_files_to_mosaic:
            st.error("Aucun fichier n'a pu être ouvert.")
            return False

        # Fusionner les rasters
        with st.spinner("Fusion des fichiers en cours..."):
            # Utiliser le bon module pour Resampling
            mosaic, out_trans = merge(src_files_to_mosaic, resampling=getattr(Resampling, methode_reechantillonnage))

            # Copier les métadonnées du premier fichier
            out_meta = src_files_to_mosaic[0].meta.copy()

            # Mettre à jour les métadonnées
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            })

            # Écrire le fichier fusionné
            with rasterio.open(fichier_sortie, "w", **out_meta) as dest:
                dest.write(mosaic)

        st.success(f"Fusion effectuée ! Fichier sauvegardé: {fichier_sortie}")

        # Fermer les fichiers
        for src in src_files_to_mosaic:
            src.close()

        return True

    except Exception as e:
        st.error(f"Erreur lors de la fusion: {str(e)}")
        return False


# Sidebar pour les options
st.sidebar.header("Options")

# Choix de la méthode de rééchantillonnage
methode_reechantillonnage = st.sidebar.selectbox(
    "Méthode de rééchantillonnage",
    options=["nearest", "bilinear", "cubic", "cubicspline", "lanczos"],
    index=1,  # Par défaut: bilinear
    help="Choisis la méthode de rééchantillonnage pour la fusion des DEMs"
)

# Upload des fichiers
st.subheader("1. Sélectionne les fichiers DEM à fusionner")
fichiers_uploaded = st.file_uploader(
    "Choisis les fichiers DEM (.hgt, .tif, etc.)",
    type=["hgt", "tif", "tiff"],
    accept_multiple_files=True
)

# Nom du fichier de sortie
st.subheader("2. Donne un nom au fichier de sortie")
nom_fichier_sortie = st.text_input(
    "Nom du fichier de sortie (sans extension)",
    value="dem_fusionnés"
)

# Extension du fichier de sortie
format_sortie = st.selectbox(
    "Format du fichier de sortie",
    options=["GeoTIFF (.tif)"],
    index=0
)

# Bouton pour lancer la fusion
st.subheader("3. Lancer la fusion")
if st.button("Fusionner les DEMs"):
    if not fichiers_uploaded:
        st.warning("Sélectionne au moins deux fichiers DEM à fusionner.")
    elif len(fichiers_uploaded) < 2:
        st.warning("Sélectionne au moins deux fichiers DEM à fusionner.")
    else:
        # Créer un répertoire temporaire pour sauvegarder les fichiers uploadés
        with tempfile.TemporaryDirectory() as temp_dir:
            fichiers_temp = []
            for file in fichiers_uploaded:
                temp_file_path = os.path.join(temp_dir, file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(file.getbuffer())
                fichiers_temp.append(temp_file_path)

            # Définir le chemin de sortie
            if format_sortie == "GeoTIFF (.tif)":
                extension = ".tif"

            fichier_sortie = os.path.join(temp_dir, f"{nom_fichier_sortie}{extension}")

            # Lancer la fusion
            st.info("Démarrage de la fusion...")
            success = fusion_dem(fichiers_temp, fichier_sortie, methode_reechantillonnage)

            if success:
                # Permettre le téléchargement du fichier fusionné
                with open(fichier_sortie, "rb") as file:
                    st.download_button(
                        label="Télécharger le DEM fusionné",
                        data=file,
                        file_name=f"{nom_fichier_sortie}{extension}",
                        mime="application/octet-stream"
                    )

# Informations supplémentaires
st.sidebar.markdown("---")
st.sidebar.subheader("À propos")
st.sidebar.info("""
Cette application permet de fusionner plusieurs fichiers DEM
en un seul fichier.
""")

# Instructions d'utilisation
with st.expander("Comment utiliser l'application ?"):
    st.markdown("""
    1. **Sélectionne les fichiers DEM** à fusionner en utilisant le champ de téléchargement.
    2. **Choisis une méthode de rééchantillonnage** dans la barre latérale :
       - `nearest` : Plus rapide mais moins précis
       - `bilinear` (par défaut) : Compromis entre vitesse et précision
       - `cubic`, `cubicspline`, `lanczos` : Plus précis mais plus lent
    3. **Donne un nom au fichier de sortie** et sélectionne le format.
    4. **Clique sur le bouton "Fusionner les DEMs"** pour lancer la fusion.
    5. Une fois la fusion terminée, **télécharge le résultat** en cliquant sur le bouton.
    """)