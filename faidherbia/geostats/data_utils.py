import os
import numpy as np

def is_mansour():
    #Test based on the expected arborescence of our respective computers
    if os.path.isdir('/home/rfernandez'):
        return False
    else:
        return True

# This function returns the path to the main directory for accessing input data and results
# 
def get_main_directory():
    if is_mansour():
        return '/Users/mansourdiene/Desktop/These/article/paper2/'
    else:
        return '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/'


# Get the acquisition dates of the aerial images
# The results can be used to access the data (i.e. can be found on disk under dir/year/full_date )
def get_aerial_imaging_dates(it_is_just_a_test=False):
    if(it_is_just_a_test ):
        full_dates=np.array(["2021_08_05"])
    else:
        #To write : full_dates=np.array([ write the dates here ])
        full_dates=None
    #Copy the array dates, but with keeping only the years
    years=np.array([date[:4] for date in full_dates])
    return full_dates,years


def get_parcel_list(it_is_just_a_test=False):
    if(it_is_just_a_test):
        return np.array(["P01"])
    else:
        return np.array(["P01","P02","P04","P05","P08","P09","P10","P11"])

read
dir = os.chdir('/Users/mansourdiene/Desktop/These/')
faidh = gpd.read_file('../These/article/paper2/data/2021/2021_08_05/shapefile/P01/faidh/faidh.shp')
crop = gpd.read_file('../These/article/paper2/data/2021/2021_08_05/shapefile/P01/limites/P01.shp')

for i in range(0, len(regions_polys)):
    #file = "P04_regions"
    plot = rio.open('../These/article/paper2/data/2021/2021_08_05/raster/P01.tif')

write
Ms_img = rio.open('/Users/mansourdiene/Desktop/These/article/paper2/result/2021/'+parc+'regions'+str(i)+'.tif','w',driver='Gtiff',
    plot = rio.open('../These/article/paper2/data/2021/2021_08_05/raster/'+parc+'.tif')
    # import vector, calculate the centroid and voronoi diagram
    faidh = gpd.read_file('../These/article/paper2/data/2021/2021_08_05/shapefile/'+parc+'/faidh/faidh.shp')
    crop = gpd.read_file('../These/article/paper2/data/2021/2021_08_05/shapefile/'+parc+'/limites/'+parc+'.shp')


