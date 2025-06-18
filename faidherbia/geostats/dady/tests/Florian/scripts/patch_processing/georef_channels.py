import rasterio
from rasterio.transform import Affine
import os
# Chemins vers les fichiers
source_file = "D:/Mes Donnees/dady_data_1/UC3_Mada/template_tif_MS_with_metadata_at_2024-02-27.tif"
# target_file = "C:/Users/fdubois/Desktop/code/data/UC3/channels_uc3/2024_2_27_Andrano_channel_3.tif"
# target_file = "D:/Mes Donnees/dady_data_1/UC3_Mada/Kickoff_dataset/Channels/2024_4_24_Andrano_channel_1.tif"
# output_file = "tests/zoo/output_georef_4_24.tif"

# Dossier contenant les .tif non géoréférencés
input_folder = "D:/Mes Donnees/dady_data_1/UC3_Mada/Kickoff_dataset/Channels"
# Dossier de sortie
output_folder = "C:/Users/fdubois/Desktop/code/data/UC3/channels_georeferences"

# Décalage en pixels
offset_x_pixels = 3160
offset_y_pixels = 2250

# Lire géoréférencement du fichier source
with rasterio.open(source_file) as src:
    transform = src.transform
    crs = src.crs

# Calcul de la nouvelle transform avec décalage
# transform = Affine(pixel_width, 0, top_left_x, 0, pixel_height, top_left_y)
new_transform = transform * Affine.translation(offset_x_pixels, offset_y_pixels)

# Parcourir tous les .tif du dossier input
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".tif"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with rasterio.open(input_path) as src:
            data = src.read()
            dtype = data.dtype
            count = data.shape[0]
            height = data.shape[1]
            width = data.shape[2]

        # Écrire le fichier avec la géoréférence et le décalage
        with rasterio.open(output_path, 'w',
                           driver='GTiff',
                           height=height,
                           width=width,
                           count=count,
                           dtype=dtype,
                           crs=crs,
                           transform=new_transform) as dst:
            dst.write(data)

        print(f"✅ {filename} traité et sauvegardé dans {output_path}")

print("🚀 Traitement terminé pour tous les fichiers.")