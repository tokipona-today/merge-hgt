import os
import streamlit as st
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import tempfile
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image
import base64

# Configuration de la page
st.set_page_config(
    page_title="Fusion de DEM",
    page_icon="🏔️",
    layout="wide"
)

# Définition des palettes de couleurs (globale pour éviter les erreurs)
colorscale_options = ["Portland", "Viridis", "Earth", "Jet", "YlGnBu", "Picnic", "RdBu", "Blackbody", "Bluered",
                      "Rainbow", "Turbo"]

# Initialiser les variables de session si elles n'existent pas
if 'fusion_done' not in st.session_state:
    st.session_state.fusion_done = False
if 'elevation_data' not in st.session_state:
    st.session_state.elevation_data = None
if 'output_transform' not in st.session_state:
    st.session_state.output_transform = None
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

st.title("Fusion de fichiers DEM (.hgt)")
st.write("Cette application permet de fusionner plusieurs fichiers DEM (Digital Elevation Model) en un seul fichier.")


@st.cache_data
def ensure_square_data(data, fill_value=None):
    """
    S'assure que les données sont carrées (même nombre de lignes et de colonnes).
    Si ce n'est pas le cas, remplit avec fill_value ou rogne selon le besoin.
    """
    height, width = data.shape

    if height == width:
        return data  # Déjà carré

    # Déterminer la nouvelle taille (carré)
    target_size = max(height, width)

    if fill_value is None:
        # Utiliser la valeur médiane comme valeur de remplissage par défaut
        fill_value = np.median(data)

    # Créer un nouveau tableau carré
    square_data = np.full((target_size, target_size), fill_value, dtype=data.dtype)

    # Calculer les offsets pour centrer les données originales
    y_offset = (target_size - height) // 2
    x_offset = (target_size - width) // 2

    # Copier les données originales dans le tableau carré
    square_data[y_offset:y_offset + height, x_offset:x_offset + width] = data

    return square_data


@st.cache_data
def subsample_data(data, sampling_factor=0.25):
    """Sous-échantillonne les données pour réduire la taille des visualisations"""
    # Calculer le pas d'échantillonnage (inverse du facteur)
    step = max(1, int(1 / sampling_factor))
    # Appliquer le sous-échantillonnage
    return data[::step, ::step]


@st.cache_data
def preview_dem_plotly(raster_data, transform, min_val=None, max_val=None, colorscale='Portland', sampling_factor=0.25):
    """Crée une prévisualisation du DEM avec Plotly"""
    # Extraire les données d'élévation
    if isinstance(raster_data, np.ndarray) and raster_data.ndim > 2:
        elevation_data = raster_data[0]  # Premier canal pour les données multibandes
    else:
        elevation_data = raster_data

    # S'assurer que les données sont carrées
    elevation_data = ensure_square_data(elevation_data)

    # Sous-échantillonner les données pour réduire la taille
    elevation_data = subsample_data(elevation_data, sampling_factor)

    # Déterminer les plages d'altitude si non fournies
    if min_val is None:
        min_val = np.percentile(elevation_data[elevation_data != elevation_data.min()], 2)
    if max_val is None:
        max_val = np.percentile(elevation_data[elevation_data != elevation_data.max()], 98)

    # Créer la figure avec Plotly
    fig = go.Figure(data=go.Heatmap(
        z=elevation_data,
        colorscale=colorscale,
        zmin=min_val,
        zmax=max_val
    ))

    # Assurer un format carré avec une taille plus grande
    fig.update_layout(
        title='Prévisualisation du DEM',
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )

    # Supprimer les axes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


@st.cache_data
def create_hillshade(elevation, azimuth=315, altitude=45):
    """Crée un ombrage de relief à partir des données d'élévation"""
    azimuth = 360.0 - azimuth
    azimuth_rad = azimuth * np.pi / 180.0
    altitude_rad = altitude * np.pi / 180.0

    # Calculer les gradients
    dx, dy = np.gradient(elevation)

    # Calculer l'ombrage
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    # Calculer l'ombrage
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)

    # Normaliser
    shaded = (shaded + 1) / 2

    return shaded


@st.cache_data
def display_hillshade_plotly(hillshade, sampling_factor=0.25):
    """Affiche l'ombrage de relief avec Plotly"""
    # S'assurer que les données sont carrées
    hillshade = ensure_square_data(hillshade)

    # Sous-échantillonner les données
    hillshade_subsampled = subsample_data(hillshade, sampling_factor)

    fig = go.Figure(data=go.Heatmap(
        z=hillshade_subsampled,
        colorscale='gray',
        showscale=False
    ))

    # Assurer un format carré avec une taille plus grande
    fig.update_layout(
        title='Ombrage de relief',
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )

    # Supprimer les axes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


@st.cache_data
def create_combined_view_plotly(elevation_data, hillshade, min_val, max_val, colorscale='Portland',
                                sampling_factor=0.25):
    """Crée une visualisation combinée avec Plotly"""
    # S'assurer que les données sont carrées
    elevation_data = ensure_square_data(elevation_data)
    hillshade = ensure_square_data(hillshade)

    # Sous-échantillonner les données
    elevation_subsampled = subsample_data(elevation_data, sampling_factor)
    hillshade_subsampled = subsample_data(hillshade, sampling_factor)

    # Créer une figure
    fig = go.Figure()

    # Ajouter la couche d'élévation
    fig.add_trace(
        go.Heatmap(
            z=elevation_subsampled,
            colorscale=colorscale,
            zmin=min_val,
            zmax=max_val,
            name='Élévation'
        )
    )

    # Ajouter la couche d'ombrage avec un mode de mélange
    # Utiliser opacity pour simuler le mélange des couches
    fig.add_trace(
        go.Heatmap(
            z=hillshade_subsampled,
            colorscale=[[0, 'rgba(0,0,0,0.5)'], [1, 'rgba(255,255,255,0)']],
            showscale=False,
            opacity=0.7,
            name='Ombrage'
        )
    )

    # Assurer un format carré avec une taille plus grande
    fig.update_layout(
        title='Couleur + ombrage',
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        )
    )

    # Supprimer les axes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


@st.cache_data
def create_3d_surface(elevation_data, min_val=None, max_val=None, sampling=20, colorscale='Portland',
                      vertical_exaggeration=0.2):
    """Crée une surface 3D interactive avec Plotly"""
    # S'assurer que les données sont carrées
    elevation_data = ensure_square_data(elevation_data)

    # Sous-échantillonner plus agressivement pour éviter les problèmes de performance et de taille
    elevation_subsampled = elevation_data[::sampling, ::sampling]

    # Déterminer les plages d'altitude si non fournies
    if min_val is None:
        min_val = np.percentile(elevation_subsampled[elevation_subsampled != elevation_subsampled.min()], 2)
    if max_val is None:
        max_val = np.percentile(elevation_subsampled[elevation_subsampled != elevation_subsampled.max()], 98)

    # Limiter encore plus la taille si nécessaire
    max_size = 100  # Taille maximale pour chaque dimension
    y_dim, x_dim = elevation_subsampled.shape
    if x_dim > max_size or y_dim > max_size:
        additional_sampling = max(1, max(x_dim, y_dim) // max_size)
        elevation_subsampled = elevation_subsampled[::additional_sampling, ::additional_sampling]
        y_dim, x_dim = elevation_subsampled.shape

    # S'assurer que les dimensions sont égales même après échantillonnage
    elevation_subsampled = ensure_square_data(elevation_subsampled)
    y_dim, x_dim = elevation_subsampled.shape  # Mettre à jour les dimensions

    # Créer une grille de coordonnées
    x = np.linspace(0, x_dim, x_dim)
    y = np.linspace(0, y_dim, y_dim)

    # Calculer une taille de contour raisonnable, en s'assurant qu'elle est positive
    contour_size = max(1.0, (max_val - min_val) / 20)

    # Créer la figure 3D
    fig = go.Figure(data=[go.Surface(
        z=elevation_subsampled,
        x=x, y=y,
        colorscale=colorscale,
        cmin=min_val,
        cmax=max_val,
        contours={
            "z": {"show": True, "start": min_val, "end": max_val, "size": contour_size}
        }
    )])

    # Calculer le rapport d'aspect approprié pour réduire l'exagération verticale
    # Plus la valeur de vertical_exaggeration est basse, moins l'altitude sera exagérée
    fig.update_layout(
        title='Vue 3D du terrain',
        autosize=False,
        width=800,
        height=800,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=vertical_exaggeration),
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Élévation (m)')
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    return fig


def fusion_dem(fichiers_entree, fichier_sortie, methode_reechantillonnage="bilinear",
               upscale_factor=1.0, apply_mask=False, min_altitude=0, max_altitude=10000):
    try:
        # Vérifier que tous les fichiers existent
        for fichier in fichiers_entree:
            if not os.path.exists(fichier):
                st.error(f"Erreur: Le fichier {fichier} n'existe pas.")
                return False, None, None

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
                return False, None, None

        if not src_files_to_mosaic:
            st.error("Aucun fichier n'a pu être ouvert.")
            return False, None, None

        # Rééchantillonnage si le facteur est différent de 1.0
        resampled_datasets = []
        if upscale_factor != 1.0:
            st.info(f"Application du facteur de rééchantillonnage: {upscale_factor}")

            for src in src_files_to_mosaic:
                # Calculer les nouvelles dimensions
                new_height = int(src.height * upscale_factor)
                new_width = int(src.width * upscale_factor)

                # Créer un temporaire
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                    tmp_file = tmp.name

                # Rééchantillonner le dataset
                with rasterio.open(tmp_file, 'w',
                                   driver='GTiff',
                                   height=new_height,
                                   width=new_width,
                                   count=src.count,
                                   dtype=src.dtypes[0],
                                   crs=src.crs,
                                   transform=src.transform * src.transform.scale(
                                       (src.width / new_width),
                                       (src.height / new_height))) as dst:
                    data = src.read(
                        out_shape=(src.count, new_height, new_width),
                        resampling=getattr(Resampling, methode_reechantillonnage)
                    )

                    dst.write(data)

                # Ouvrir le fichier rééchantillonné
                resampled_src = rasterio.open(tmp_file)
                resampled_datasets.append(resampled_src)

            # Utiliser les fichiers rééchantillonnés si disponibles
            if resampled_datasets:
                # Fermer les fichiers originaux
                for src in src_files_to_mosaic:
                    src.close()
                src_files_to_mosaic = resampled_datasets

        # Fusionner les rasters
        with st.spinner("Fusion des fichiers en cours..."):
            # Utiliser le bon module pour Resampling
            mosaic, out_trans = merge(src_files_to_mosaic, resampling=getattr(Resampling, methode_reechantillonnage))

            # Déterminer la valeur NoData
            nodata_value = src_files_to_mosaic[0].nodata if src_files_to_mosaic[0].nodata is not None else -9999

            # Déterminer l'altitude minimale valide dans les données
            # Créer un masque pour les valeurs non-NoData
            valid_mask = mosaic[0] != nodata_value
            if np.any(valid_mask):
                # Trouver l'altitude minimale valide en excluant les valeurs aberrantes
                # (comme les -32768 ou autres valeurs très négatives souvent utilisées pour NoData)
                valid_elevations = mosaic[0][valid_mask]
                # Utiliser le 1er percentile pour éviter les outliers
                min_valid_elevation = np.percentile(valid_elevations[valid_elevations > -1000], 1)
                st.info(f"Altitude minimale valide détectée: {min_valid_elevation:.2f}m")

                # Remplacer les valeurs NoData par l'altitude minimale valide
                mosaic[0] = np.where(mosaic[0] == nodata_value, min_valid_elevation, mosaic[0])

                # Remplacer également toutes les valeurs aberrantes (trop négatives)
                mosaic[0] = np.where(mosaic[0] < -1000, min_valid_elevation, mosaic[0])
            else:
                st.warning("Aucune valeur d'élévation valide trouvée. Utilisation de 0 comme valeur par défaut.")
                mosaic[0] = np.where(mosaic[0] == nodata_value, 0, mosaic[0])

            # Appliquer un masque si demandé
            if apply_mask:
                st.info(f"Application du masque d'altitude: {min_altitude}m - {max_altitude}m")
                # Créer un masque basé sur l'altitude
                mask = (mosaic[0] >= min_altitude) & (mosaic[0] <= max_altitude)
                # Appliquer le masque
                mosaic[0] = np.where(mask, mosaic[0], min_valid_elevation)

            # S'assurer que les données sont carrées pour l'exportation
            height, width = mosaic[0].shape
            st.info(f"Dimensions originales du DEM fusionné: {width}x{height}")

            if height != width:
                st.info(f"Conversion du DEM en format carré pour l'exportation")
                # Utiliser ensure_square_data pour rendre le DEM carré
                square_data = ensure_square_data(mosaic[0],
                                                 fill_value=min_valid_elevation if 'min_valid_elevation' in locals() else 0)

                # Créer un nouveau tableau mosaic avec les dimensions correctes
                new_height, new_width = square_data.shape
                st.info(f"Nouvelles dimensions carrées: {new_width}x{new_height}")

                # Recréer le tableau mosaic au lieu de simplement assigner à mosaic[0]
                new_mosaic = np.zeros((1, new_height, new_width), dtype=mosaic.dtype)
                new_mosaic[0] = square_data
                mosaic = new_mosaic

                # Calculer la nouvelle transformation pour le dataset carré
                pixel_width = out_trans[0]  # Taille du pixel en X
                pixel_height = out_trans[5]  # Taille du pixel en Y (généralement négative)

                # Déterminer le décalage pour centrer les données
                x_offset = (new_width - width) // 2
                y_offset = (new_height - height) // 2

                # Calculer la nouvelle origine pour que les données originales restent alignées
                new_x_origin = out_trans[2] - x_offset * pixel_width
                new_y_origin = out_trans[4] - y_offset * pixel_height

                # Créer la nouvelle transformation
                from rasterio.transform import Affine
                out_trans = Affine(pixel_width, out_trans[1], new_x_origin,
                                   out_trans[3], pixel_height, new_y_origin)

            # Copier les métadonnées du premier fichier
            out_meta = src_files_to_mosaic[0].meta.copy()

            # Mettre à jour les métadonnées
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            })

            # Obtenir l'extension du fichier de sortie
            _, ext = os.path.splitext(fichier_sortie)

            # Si le format est HGT, on sauvegarde en raw binary
            if ext.lower() == '.hgt':
                # Obtenir les données d'élévation
                elevation_data = mosaic[0]  # Prendre le premier canal

                # Vérifier/conserver les dimensions standard pour HGT
                height, width = elevation_data.shape
                st.info(f"Dimensions du fichier HGT: {width}x{height}")

                # Convertir en entiers signés sur 16 bits (format HGT standard)
                elevation_int16 = elevation_data.astype(np.int16)

                # Le format HGT utilise big-endian (réseau)
                elevation_bytes = elevation_int16.byteswap().tobytes()

                # Écrire le fichier HGT (raw binary)
                with open(fichier_sortie, 'wb') as hgt_file:
                    hgt_file.write(elevation_bytes)

                # Créer un fichier world (.hgw) pour maintenir la géoréférence
                world_file_path = os.path.splitext(fichier_sortie)[0] + '.hgw'
                with open(world_file_path, 'w') as wf:
                    pixel_width = out_trans[0]
                    rotation_1 = out_trans[1]
                    rotation_2 = out_trans[3]
                    pixel_height = out_trans[5]
                    upper_left_x = out_trans[2]
                    upper_left_y = out_trans[4]
                    wf.write(
                        f"{pixel_width}\n{rotation_1}\n{rotation_2}\n{pixel_height}\n{upper_left_x}\n{upper_left_y}")

                st.success(f"Fusion effectuée ! Fichier HGT sauvegardé: {fichier_sortie}")
                st.info(f"Un fichier world (.hgw) a été créé pour conserver la géoréférence")
            else:
                # Écrire le fichier fusionné au format GeoTIFF
                with rasterio.open(fichier_sortie, "w", **out_meta) as dest:
                    dest.write(mosaic)

                st.success(f"Fusion effectuée ! Fichier sauvegardé: {fichier_sortie}")

            # Nettoyer les fichiers temporaires
            if resampled_datasets:
                for src in resampled_datasets:
                    src_path = src.name
                    src.close()
                    if os.path.exists(src_path):
                        os.remove(src_path)
            else:
                # Fermer les fichiers
                for src in src_files_to_mosaic:
                    src.close()

            return True, mosaic, out_trans

    except Exception as e:
        st.error(f"Erreur lors de la fusion: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False, None, None


# Fonction pour créer le répertoire temporaire
def create_temp_dir():
    # Si un répertoire temporaire existe déjà, on le supprime
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            import shutil
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass

    # Créer un nouveau répertoire temporaire
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir
    return temp_dir


# Fonction pour gérer la fusion
def process_fusion():
    # Réinitialiser l'état de fusion
    st.session_state.fusion_done = False
    st.session_state.elevation_data = None
    st.session_state.output_transform = None
    st.session_state.output_file = None

    if not fichiers_uploaded:
        st.sidebar.warning("Sélectionne au moins deux fichiers DEM à fusionner.")
        return
    elif len(fichiers_uploaded) < 2:
        st.sidebar.warning("Sélectionne au moins deux fichiers DEM à fusionner.")
        return

    # Créer un répertoire temporaire
    temp_dir = create_temp_dir()

    fichiers_temp = []
    for file in fichiers_uploaded:
        temp_file_path = os.path.join(temp_dir, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())
        fichiers_temp.append(temp_file_path)

    # Définir le chemin de sortie et l'extension
    if format_sortie == "GeoTIFF (.tif)":
        extension = ".tif"
    elif format_sortie == "SRTM HGT (.hgt)":
        extension = ".hgt"

    fichier_sortie = os.path.join(temp_dir, f"{nom_fichier_sortie}{extension}")

    # Lancer la fusion
    st.info("Démarrage de la fusion...")
    success, mosaic, out_trans = fusion_dem(
        fichiers_temp,
        fichier_sortie,
        methode_reechantillonnage,
        upscale_factor,
        apply_mask,
        min_altitude,
        max_altitude
    )

    if success and mosaic is not None:
        # Sauvegarder les résultats dans la session
        st.session_state.fusion_done = True
        st.session_state.elevation_data = mosaic[0]
        st.session_state.output_transform = out_trans
        st.session_state.output_file = fichier_sortie

        # Recharger la page pour afficher les résultats
        st.rerun()


# Configuration de la barre latérale
# Sidebar pour les options
st.sidebar.header("Options de fusion")

# Upload des fichiers
st.sidebar.subheader("1. Sélectionne les fichiers DEM à fusionner")
fichiers_uploaded = st.sidebar.file_uploader(
    "Choisis les fichiers DEM (.hgt, .tif, etc.)",
    type=["hgt", "tif", "tiff"],
    accept_multiple_files=True
)

# Nom du fichier de sortie
st.sidebar.subheader("2. Donne un nom au fichier de sortie")
nom_fichier_sortie = st.sidebar.text_input(
    "Nom du fichier de sortie (sans extension)",
    value="dem_fusionnés"
)

# Extension du fichier de sortie
format_sortie = st.sidebar.selectbox(
    "Format du fichier de sortie",
    options=["GeoTIFF (.tif)", "SRTM HGT (.hgt)"],
    index=0
)

# Choix de la méthode de rééchantillonnage
methode_reechantillonnage = st.sidebar.selectbox(
    "Méthode de rééchantillonnage",
    options=["nearest", "bilinear", "cubic", "cubicspline", "lanczos"],
    index=1,  # Par défaut: bilinear
    help="Choisis la méthode de rééchantillonnage pour la fusion des DEMs"
)

# Options avancées
with st.sidebar.expander("Options avancées"):
    upscale_factor = st.slider(
        "Facteur de rééchantillonnage (résolution)",
        min_value=0.25,
        max_value=4.0,
        value=1.0,
        step=0.25,
        help="Modifier la résolution du fichier de sortie (< 1 = réduction, > 1 = augmentation)"
    )

    # Options de masquage
    apply_mask = st.checkbox("Appliquer un masque d'altitude", value=False)
    if apply_mask:
        min_altitude = st.number_input("Altitude minimale (masquer en dessous)", value=0)
        max_altitude = st.number_input("Altitude maximale (masquer au-dessus)", value=10000)
    else:
        min_altitude = 0
        max_altitude = 10000

# Bouton pour lancer la fusion
st.sidebar.subheader("3. Lancer la fusion")
if st.sidebar.button("Fusionner les DEMs"):
    process_fusion()

# Si la fusion a été effectuée, afficher le bouton de téléchargement et les contrôles de visualisation
if st.session_state.fusion_done and st.session_state.elevation_data is not None:
    # Mettre en évidence le bouton de téléchargement
    st.sidebar.markdown("---")
    st.sidebar.subheader("Télécharger le résultat")

    # Obtenir les données d'élévation
    elevation_data = st.session_state.elevation_data

    # Ajouter une option de sous-échantillonnage pour l'export
    export_sampling = st.sidebar.select_slider(
        "Sous-échantillonnage pour l'export",
        options=[1.0, 0.75, 0.5, 0.25],
        value=1.0,
        help="Réduire la taille du fichier exporté (1.0 = taille originale, 0.5 = 25% de la taille originale)"
    )

    # Créer un bouton de téléchargement plus visible
    if export_sampling < 1.0:
        # Si sous-échantillonnage demandé, créer une version réduite du fichier
        # Utiliser st.spinner au lieu de st.sidebar.spinner
        with st.spinner(f"Préparation du fichier exporté ({int(export_sampling * 100)}% de la taille originale)..."):
            # Ouvrir le fichier original
            with rasterio.open(st.session_state.output_file) as src:
                # Extraire les données
                data = src.read(1)  # Lire la première bande
                profile = src.profile.copy()

                # Calculer les nouvelles dimensions
                orig_height, orig_width = data.shape
                new_height = int(orig_height * export_sampling)
                new_width = int(orig_width * export_sampling)

                # Confirmer que les dimensions restent carrées
                if new_height != new_width:
                    new_size = max(new_height, new_width)
                    new_height = new_size
                    new_width = new_size

                # Sous-échantillonner les données
                step = int(1 / export_sampling)
                resampled_data = data[::step, ::step]

                # Ajuster les dimensions si nécessaire (pour garantir qu'elles sont carrées)
                if resampled_data.shape[0] != resampled_data.shape[1]:
                    resampled_data = ensure_square_data(resampled_data)

                # Préparer un fichier temporaire
                with tempfile.NamedTemporaryFile(suffix='.tif' if profile['driver'] == 'GTiff' else '.hgt',
                                                 delete=False) as tmp:
                    export_file = tmp.name

                # Mettre à jour le profil
                profile.update(
                    height=resampled_data.shape[0],
                    width=resampled_data.shape[1],
                    transform=src.transform * src.transform.scale(
                        (src.width / resampled_data.shape[1]),
                        (src.height / resampled_data.shape[0])
                    )
                )

                # Écrire le fichier réduit
                with rasterio.open(export_file, 'w', **profile) as dst:
                    dst.write(resampled_data, 1)

        # Télécharger la version réduite
        with open(export_file, "rb") as file:
            btn = st.sidebar.download_button(
                label=f"📥 Télécharger le DEM fusionné ({int(export_sampling * 100)}%)",
                data=file,
                file_name=f"{nom_fichier_sortie}_reduced_{int(export_sampling * 100)}{'.tif' if format_sortie == 'GeoTIFF (.tif)' else '.hgt'}",
                mime="application/octet-stream",
                use_container_width=True,
                type="primary"  # Utiliser un bouton primaire pour le mettre en évidence
            )
    else:
        # Télécharger la version originale
        with open(st.session_state.output_file, "rb") as file:
            btn = st.sidebar.download_button(
                label=f"📥 Télécharger le DEM fusionné",
                data=file,
                file_name=f"{nom_fichier_sortie}{'.tif' if format_sortie == 'GeoTIFF (.tif)' else '.hgt'}",
                mime="application/octet-stream",
                use_container_width=True,
                type="primary"  # Utiliser un bouton primaire pour le mettre en évidence
            )

    # Afficher les statistiques dans un expander
    with st.sidebar.expander("Statistiques du DEM", expanded=True):
        # Statistiques de base
        stats = {
            "Altitude minimale": float(np.min(elevation_data)),
            "Altitude maximale": float(np.max(elevation_data)),
            "Altitude moyenne": float(np.mean(elevation_data)),
            "Écart type": float(np.std(elevation_data)),
            "Nombre de pixels": int(np.prod(elevation_data.shape))
        }
        st.json(stats)

        # Histogramme des élévations avec Plotly (sous-échantillonnage pour réduire la taille)
        # Sous-échantillonner les données pour l'histogramme
        sample_size = min(100000, elevation_data.size)  # Limiter à 100K points
        indices = np.random.choice(elevation_data.size, size=sample_size, replace=False)
        sampled_data = elevation_data.flatten()[indices]

        fig = go.Figure(data=[go.Histogram(x=sampled_data, nbinsx=100)])
        fig.update_layout(
            title="Distribution des altitudes (échantillon)",
            xaxis_title="Altitude (m)",
            yaxis_title="Nombre de pixels",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # Paramètres de visualisation dans un SEUL expander
    with st.sidebar.expander("Paramètres visuels", expanded=False):
        # Mise à jour des palettes de couleurs pour Plotly - une seule palette pour toutes les vues
        colorscale = st.selectbox(
            "Palette de couleurs",
            options=colorscale_options,
            index=0,  # Portland par défaut
            key="colorscale_viz"  # Clé unique
        )

        # Paramètres de niveau
        st.subheader("Ajustement des niveaux")
        min_elev = st.slider(
            "Altitude minimale pour l'affichage",
            min_value=float(np.min(elevation_data)),
            max_value=float(np.max(elevation_data)),
            value=float(np.percentile(elevation_data[elevation_data != elevation_data.min()], 2)),
            format="%.1f",
            key="min_elev_viz"  # Clé unique
        )
        max_elev = st.slider(
            "Altitude maximale pour l'affichage",
            min_value=float(np.min(elevation_data)),
            max_value=float(np.max(elevation_data)),
            value=float(np.percentile(elevation_data[elevation_data != elevation_data.max()], 98)),
            format="%.1f",
            key="max_elev_viz"  # Clé unique
        )

        # Contrôles pour l'ombrage
        st.subheader("Paramètres d'ombrage")
        azimuth = st.slider("Azimuth (direction de la lumière)", 0, 360, 315, key="azimuth_viz")  # Clé unique
        altitude = st.slider("Altitude (hauteur de la lumière)", 0, 90, 45, key="altitude_viz")  # Clé unique

        # Paramètres 3D uniquement si la vue 3D est sélectionnée
        if "selected_viz" in locals() and viz_options[selected_viz] == "3d":
            st.subheader("Paramètres 3D")
            sampling_rate = st.slider(
                "Taux de sous-échantillonnage",
                min_value=5,
                max_value=50,
                value=20,
                key="sampling_3d_viz"  # Clé unique
            )

            vertical_exaggeration = st.slider(
                "Facteur d'exagération verticale",
                min_value=0.05,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Contrôle l'accentuation de l'altitude (valeur faible = relief moins accentué)",
                key="vertical_exaggeration_viz"  # Clé unique
            )

        # Facteur de sous-échantillonnage pour toutes les visualisations 2D
        st.subheader("Performance")
        viz_sampling_factor = st.select_slider(
            "Facteur de sous-échantillonnage pour visualisations 2D",
            options=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            value=0.25,
            key="viz_sampling_factor_viz",  # Clé unique
            help="Réduire cette valeur si vous rencontrez des problèmes de mémoire"
        )


# Informations supplémentaires (toujours en bas de la sidebar)
st.sidebar.markdown("---")
sidebar_bottom_container = st.sidebar.container()

with sidebar_bottom_container.expander("À propos", expanded=False):
    st.info("""
    Cette application permet de fusionner plusieurs fichiers DEM
    en un seul fichier et de réaliser diverses visualisations.
    """)

# Instructions d'utilisation
with sidebar_bottom_container.expander("Comment utiliser l'application ?", expanded=False):
    st.markdown("""
    1. **Sélectionne les fichiers DEM** à fusionner.
    2. **Choisis une méthode de rééchantillonnage**:
       - `nearest` : Plus rapide mais moins précis
       - `bilinear` (par défaut) : Compromis entre vitesse et précision
       - `cubic`, `cubicspline`, `lanczos` : Plus précis mais plus lent
    3. **Explore les options avancées**:
       - Facteur de rééchantillonnage pour ajuster la résolution
       - Masquage d'altitude pour filtrer certaines zones
    4. **Clique sur "Fusionner les DEMs"** pour lancer la fusion.
    5. Ajuste les paramètres visuels dans les commandes sur la gauche.
    """)


    # Si la fusion a été effectuée, afficher les résultats
if st.session_state.fusion_done and st.session_state.elevation_data is not None:
    # Section principale - Menu de visualisation
    st.write("## Visualisation du DEM fusionné")

    # Menu de sélection pour choisir la visualisation
    viz_options = {
        "Ombrage de relief": "hillshade",
        "Visualisation standard": "standard",
        "Visualisation combinée": "combined",
        "Vue 3D": "3d"
    }
    selected_viz = st.radio(
        "Choisir le type de visualisation:",
        options=list(viz_options.keys()),
        horizontal=True,
        index=2  # Visualisation combinée par défaut (index 2 dans la liste)
    )

    # Obtenir les données d'élévation
    elevation_data = st.session_state.elevation_data
    out_trans = st.session_state.output_transform

    # Définir des valeurs par défaut pour les paramètres de visualisation
    # Au cas où l'utilisateur n'a pas encore ouvert l'expander des paramètres
    min_elev_default = float(np.percentile(elevation_data[elevation_data != elevation_data.min()], 2))
    max_elev_default = float(np.percentile(elevation_data[elevation_data != elevation_data.max()], 98))

    # Préparer l'ombrage seulement si nécessaire
    hillshade = None

    # Obtenir les valeurs depuis session_state ou utiliser les valeurs par défaut
    min_elev = st.session_state.get('min_elev_viz', min_elev_default)
    max_elev = st.session_state.get('max_elev_viz', max_elev_default)
    colorscale = st.session_state.get('colorscale_viz', 'Portland')
    azimuth = st.session_state.get('azimuth_viz', 315)
    altitude = st.session_state.get('altitude_viz', 45)
    viz_sampling_factor = st.session_state.get('viz_sampling_factor_viz', 0.25)
    sampling_rate = st.session_state.get('sampling_3d_viz', 20)
    vertical_exaggeration = st.session_state.get('vertical_exaggeration_viz', 0.2)

    # Afficher uniquement la visualisation sélectionnée et calculer seulement ce qui est nécessaire
    if viz_options[selected_viz] == "standard":
        fig = preview_dem_plotly(elevation_data, out_trans, min_elev, max_elev, colorscale, viz_sampling_factor)
        st.plotly_chart(fig, use_container_width=False)

    elif viz_options[selected_viz] == "hillshade":
        # Créer l'ombrage seulement si on en a besoin
        hillshade = create_hillshade(elevation_data, azimuth, altitude)
        fig = display_hillshade_plotly(hillshade, viz_sampling_factor)
        st.plotly_chart(fig, use_container_width=False)

    elif viz_options[selected_viz] == "combined":
        # Créer l'ombrage seulement si on en a besoin
        hillshade = create_hillshade(elevation_data, azimuth, altitude)
        fig = create_combined_view_plotly(elevation_data, hillshade, min_elev, max_elev, colorscale,
                                          viz_sampling_factor)
        st.plotly_chart(fig, use_container_width=False)

    elif viz_options[selected_viz] == "3d":
        with st.spinner("Génération de la vue 3D..."):
            fig_3d = create_3d_surface(elevation_data, min_elev, max_elev, sampling_rate, colorscale,
                                       vertical_exaggeration)
            st.plotly_chart(fig_3d, use_container_width=False)
