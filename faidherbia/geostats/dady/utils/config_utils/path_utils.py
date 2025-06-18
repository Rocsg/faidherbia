import os

# Base directory of the project
BASE_DIR = os.path.dirname("C:/Users/fdubois/Desktop/code")

# Dossiers importants
DATA_DIR = os.path.join(BASE_DIR, '/data')
OUTPUT_TEST_DIR = os.path.join(BASE_DIR, 'tests/zoo')
ORIGINAL_TIF = os.path.join("D:/Mes Donnees/dady_data_1/UC3_Mada/template_tif_MS_with_metadata_at_2024-02-27.tif")
CHANNELS_DIR = os.path.join(DATA_DIR, 'UC3/channels_georeferences')
CHANNELS_PATCHES_DIR = os.path.join(DATA_DIR, 'UC3/channels_uc3')
SHAPEFILES = os.path.join(DATA_DIR, 'UC3/AllBandes_shapefiles/All Bands32338.shp')