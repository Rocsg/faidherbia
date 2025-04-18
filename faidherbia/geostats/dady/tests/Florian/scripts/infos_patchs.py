raster_folder_path = "data/UC3/channels_georeferences"
shapefile_path = "data/UC3/AllBandes_shapefiles/All Bands32338.shp"

import os
from PIL import Image

# Chemin du dossier contenant les .tif
dossier = "tests/zoo/batch_rasters/2024_2_27_Andrano_channel_0"

# Dictionnaire pour compter les tailles
taille_counts = {}

# Parcours des fichiers du dossier
for fichier in os.listdir(dossier):
    if fichier.lower().endswith(".tif"):
        chemin_complet = os.path.join(dossier, fichier)
        try:
            with Image.open(chemin_complet) as img:
                taille = img.size  # (largeur, hauteur)
                taille_str = f"{taille[0]}x{taille[1]}"
                if taille_str in taille_counts:
                    taille_counts[taille_str] += 1
                else:
                    taille_counts[taille_str] = 1
        except Exception as e:
            print(f"Erreur avec le fichier {fichier} : {e}")

# Affichage du résultat
for taille, count in taille_counts.items():
    print(f"{taille} : {count}")
