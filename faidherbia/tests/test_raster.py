

import geopandas as gpd
import rasterio
tgt_crs = "EPSG:4326"

for i in ["01","02","04","05","08","10"]:
    print (i)
    faid_shape = gpd.read_file('/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/2021/2021_08_05/shapefile/P'+i+'/faidh/faidh.shp')
    print("crsinit=" + str(faid_shape.crs))
    if faid_shape.crs != tgt_crs:
        faid_shape = faid_shape.to_crs(tgt_crs)
    print("crsthen="+str(faid_shape.crs))
 
    parcel_shape = gpd.read_file('/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/2021/2021_08_05/shapefile/P'+i+'/limites/P'+i+'.shp')
    print("crsinit="+str(parcel_shape.crs))
    if parcel_shape.crs != tgt_crs:
        parcel_shape = parcel_shape.to_crs(tgt_crs)
    print("crsthen="+str(parcel_shape.crs))
    src= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/2021/2021_08_05/raster/P"+i+".tif")
    print("crsraster="+str(src.crs))
#    print(faid_shape.crs=='epsg:32628')    
