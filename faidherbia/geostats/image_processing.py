import numpy as np  
import rasterio as rio
import fiona
from math import sqrt,exp
import os
from rasterio import mask


# This function apply some shapefile as mask on rasters, then outputs the results of the operation
# Example : path_to_raster='/Users/macbookpro/Desktop/These/Data/NIAKHAR/Results/subplots/segmented/Plot_segmented_ms/2021_09_24'
#           path_to_shapefile='/Users/macbookpro/Desktop/These/Data/NIAKHAR/Shapefile/shapefile'
def subplot_extraction(path_to_raster,path_to_shapefile):
    dirpath = path_to_raster
    img_list =[]
    for k in os.listdir(dirpath):
        file = os.path.splitext(k)[0]
        img_list.append(file)
    img_list = sorted(img_list)
    #img_list.pop(0) 
    # TODO
    print("Length of img_list="+str(len(img_list)))

    dirpath2 = path_to_shapefile
    shp_list = []
    for f in os.listdir(dirpath2):
        file_shp = os.path.splitext(f)[0]
        shp_list.append(file_shp)
    shp_list =sorted(shp_list)
    shp_list.pop(0)
    len(shp_list)




    j = 0
    for k in img_list:
        fichier1 = k
        #ms=[] # This list will contain the mean height values of the 60 plots
        #out_imgs=[]
        with rasterio.open('/Users/macbookpro/Desktop/These/Data/NIAKHAR/Results/subplots/segmented/Plot_segmented_ms/2021_09_24/'+fichier1+'.tif','r') as chm:
            j+=1
            for f_ in shp_list:
                fichier= f_
                name = f'{fichier}' 
                with fiona.open('/Users/macbookpro/Desktop/These/Data/NIAKHAR/Shapefile/shapefile/'+fichier+'.shp') as shapefile:
                    features = [feature['geometry'] for feature in shapefile]
                    try : 
                        out_img,out_transform = mask.mask(chm,features,crop = True, filled = False)
                        #msavi2 = (2 * (nir + 1) - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red)))/2
                        if k[:3]==fichier[:3]:
                            out_meta = chm.meta.copy()
                            out_meta.update({"driver": "GTiff",
                            "height": out_img.shape[1],
                            "width": out_img.shape[2],
                            "transform": out_transform})
                            with rasterio.open(f'/Users/macbookpro/Desktop/These/Data/NIAKHAR/Results/subplots/segmented/subplots/{fichier}.tif', "w",  **out_meta) as dst:
                                dst.write(out_img)
                    except :
                        continue
    #A function for masking

    #A function for threshold

    #A function for nanmean of I don't know what


