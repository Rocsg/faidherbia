import rasterio

print( "########## info raster ##########")
# Ouvrir le fichier TIFF
with rasterio.open("D:/Mes Donnees/dady_data_1/UC3_Mada/template_tif_MS_with_metadata_at_2024-02-27.tif") as src:
    # Vérifier s'il y a un CRS défini
    if src.crs is None:
        print("Le fichier n'a pas de système de coordonnées de référence (CRS) défini.")
    else:
        print(f"CRS défini : {src.crs}")
    
    # Vérifier la transformation affine
    if src.transform.is_identity:
        print("Le fichier n'a pas de transformation géographique (utilise la matrice d'identité).")
    else:
        print(f"Transformation définie : {src.transform}")
        print(f"Origine (coin supérieur gauche) : ({src.transform.c}, {src.transform.f})")
        print(f"Taille du pixel : ({src.transform.a}, {src.transform.e})")
    
    # Afficher les limites en coordonnées
    print(f"Limites en coordonnées : {src.bounds}")

print( "########## info shapefile ##########")

import geopandas as gpd
gdf = gpd.read_file("data/UC3/AllBandes_shapefiles/All Bands.shp")
print("shapfile crs : ", gdf.crs)
print("shapefile totalbounds : ", gdf.total_bounds)
# gdf.set_crs(epsg=32338, inplace=True)
# gdf.to_file("data/UC3/AllBandes_shapefiles/All Bands_clean.shp", encoding="utf-8")
