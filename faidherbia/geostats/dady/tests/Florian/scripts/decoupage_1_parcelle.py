import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os

# Chemins
raster_path = "data/UC3/channels_georeferences/2024_2_27_Andrano_channel_0.tif"
shapefile_path = "data/UC3/AllBandes_shapefiles/All Bands32338.shp"
output_dir = "tests/zoo"


# Lire le shapefile
gdf = gpd.read_file(shapefile_path)

# Assurer la correspondance de projection
with rasterio.open(raster_path) as src:
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    # 🔹 Prendre le 1er polygone
    polygon = gdf.iloc[0]
    geom = [polygon.geometry.__geo_interface__]  # rasterio.mask attend une liste de géométries GeoJSON

    # Découpe du raster
    out_image, out_transform = mask(src, geom, crop=True)
    out_meta = src.meta.copy()

    # Mise à jour des métadonnées
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Nom de fichier de sortie (on peut utiliser un attribut du polygone, ex: polygon["id"])
    raster_filename = f"raster_poly_0.tif"
    output_path = os.path.join(output_dir, raster_filename)

    # Écriture du raster
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

# 🔍 Afficher les métadonnées du polygone pour vérification
print("✅ Raster extrait pour le 1er polygone.")
print("📄 Attributs associés au polygone :")
print(polygon)
