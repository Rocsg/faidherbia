import numpy as np
import math
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from faidherbia.geostats.polar_viewer import polar_viewer_on_actual_data
from faidherbia.geostats.geometry_utils import get_patch_of_raster_around_tree
from rasterio.warp import reproject, Resampling
import numpy as np
import rasterio

        


# Open the first raster (A)
def resample_raster_and_write(reference_path,source_path,destination_path):
    with rasterio.open(reference_path) as src_A:
        # Get the transform, pixel size, and size in pixels of raster A
        transform_A = src_A.transform
        pixel_size_A = src_A.res[0]
        width_A = src_A.width
        height_A = src_A.height

        # Open the second raster (B)
        with rasterio.open(source_path) as src_B:
            # Reproject raster B to the same resolution and extent as raster A
            data_B, transform_B = reproject(
                source=rasterio.band(src_B, 1),
                destination=np.zeros((height_A, width_A), dtype=np.float32),
                src_transform=src_B.transform,
                src_crs=src_B.crs,
                dst_transform=transform_A,
                dst_crs=src_A.crs,
                resampling=Resampling.bilinear)

            # Write the reprojected data to a new raster file
            profile_B = src_B.profile
            profile_B.update(transform=transform_A, width=width_A, height=height_A)
            with rasterio.open(destination_path, 'w', **profile_B) as dst:
                dst.write(data_B, 1)


year="2021"
date="2021_08_05"


#
#for parcel in ["P01"]:
for parcel in ["P01", "P02", "P04","P05", "P08", "P09", "P10", "P11"]:
    print("")
    print("Parcel: "+parcel)
    reference_path="/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/MS/"+parcel+".tif"
    source_path="/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/Yield/"+parcel+".tif"
    target_path="/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/Yield/"+parcel+"_res.tif"
    resample_raster_and_write(reference_path,source_path,target_path)

    source_path="/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/Biomass/"+parcel+".tif"
    target_path="/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/Biomass/"+parcel+"_res.tif"
    resample_raster_and_write(reference_path,source_path,target_path)

