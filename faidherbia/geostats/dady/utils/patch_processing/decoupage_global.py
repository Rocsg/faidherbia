import rasterio
import geopandas as gpd
from rasterio.mask import mask
import os

# Chemins
raster_path = "data/UC3/channels_georeferences/2024_2_27_Andrano_channel_0.tif"
shapefile_path = "data/UC3/AllBandes_shapefiles/All Bands32338.shp"
output_path = "tests/zoo/raster_decoupe.tif"

# Charger le shapefile
gdf = gpd.read_file(shapefile_path)
# S'assurer que le CRS est le même que le raster
with rasterio.open(raster_path) as src:
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)
    shapes = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]

    # Découper le raster avec les géométries du shapefile
    out_image, out_transform = mask(src, shapes, crop=True)
    out_meta = src.meta.copy()

# Mettre à jour les métadonnées
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

# Sauvegarder le raster découpé
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"✅ Raster découpé et sauvegardé ici : {output_path}")
