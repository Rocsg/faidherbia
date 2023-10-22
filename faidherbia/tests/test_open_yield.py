import numpy as np
import math
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from faidherbia.geostats.polar_viewer import polar_viewer_on_actual_data
from faidherbia.geostats.geometry_utils import get_patch_of_raster_around_tree
import numpy as np
import rasterio

year="2021"
date="2021_08_05"


for parcel in ["P01", "P02", "P04","P05", "P08", "P09", "P10", "P11"]:
    print("")
    print("Parcel: "+parcel)
    print("Geometry of MS raster")
    parcel_raster= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/MS/"+parcel+".tif")
    patch_size_in_meters_x=100000*parcel_raster.width*parcel_raster.transform[0]
    patch_size_in_meters_y=100000*parcel_raster.height*parcel_raster.transform[0]
    print(" - Parcel size in pixels : "+str(parcel_raster.width)+" x "+str(parcel_raster.height)+" ")
    print(" - Pixel size            : "+str(parcel_raster.transform[0]) + " (degrees ?)")
    print(" - Actual parcel size    : "+str(patch_size_in_meters_x)+" x "+str(patch_size_in_meters_y)+" ")

    centerX=parcel_raster.width/2
    centerY=parcel_raster.height/2
    window_area = rasterio.windows.Window(centerX-1500, centerY-1500, 3000, 3000)

    patch_ms = parcel_raster.read(window=window_area, masked=False)
    patch_ms=patch_ms[1,:,:]
    print(np.shape(patch_ms))
    
    #save patch_ms as tif
    from skimage.io import imsave
    imsave("/home/rfernandez/Bureau/ms.tif", patch_ms)
    


    print("Geometry of Yield raster")
    parcel_raster= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/Yield/"+parcel+"_res.tif")
    patch_size_in_meters_x=100000*parcel_raster.width*parcel_raster.transform[0]
    patch_size_in_meters_y=100000*parcel_raster.height*parcel_raster.transform[0]
    print(" - Parcel size in pixels : "+str(parcel_raster.width)+" x "+str(parcel_raster.height)+" ")
    print(" - Pixel size            : "+str(parcel_raster.transform[0])+ "degrees ?)")
    print(" - Actual parcel size    : "+str(patch_size_in_meters_x)+" x "+str(patch_size_in_meters_y)+" ")

    patch_yield = parcel_raster.read(window=window_area, masked=False)
    patch_yield=patch_yield[0,:,:]
    print(np.shape(patch_yield))
    imsave("/home/rfernandez/Bureau/yield.tif", patch_yield)
    
    time.sleep(1000)


