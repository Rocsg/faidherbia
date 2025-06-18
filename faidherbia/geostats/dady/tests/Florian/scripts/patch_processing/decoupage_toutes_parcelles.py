import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os

# Chemins
raster_folder = "data/UC3/channels_georeferences"
shapefile_path = "data/UC3/AllBandes_shapefiles/All Bands32338.shp"
output_base_dir = "tests/zoo/batch_rasters"
os.makedirs(output_base_dir, exist_ok=True)

# Lire le shapefile
gdf = gpd.read_file(shapefile_path)

# Boucle sur tous les fichiers raster
for raster_name in os.listdir(raster_folder):
    if raster_name.lower().endswith(".tif"):
        raster_path = os.path.join(raster_folder, raster_name)
        raster_base = os.path.splitext(raster_name)[0]

        # Créer un dossier spécifique pour ce raster
        raster_output_dir = os.path.join(output_base_dir, raster_base)
        os.makedirs(raster_output_dir, exist_ok=True)

        with rasterio.open(raster_path) as src:
            # Harmoniser les CRS
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            # Découper chaque polygone
            for idx, polygon in gdf.iterrows():
                geom = [polygon.geometry.__geo_interface__]

                try:
                    out_image, out_transform = mask(src, geom, crop=True)
                except Exception as e:
                    print(f"⚠️ Erreur avec {raster_name}, polygone {idx} : {e}")
                    continue

                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Utiliser l'id du polygone si dispo
                poly_band_plot = polygon.get("band_plot", idx)
                output_filename = f"{raster_base}_patch_{poly_band_plot}.tif"
                output_path = os.path.join(raster_output_dir, output_filename)

                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

                print(f"✅ {output_filename} enregistré dans {raster_output_dir}")
