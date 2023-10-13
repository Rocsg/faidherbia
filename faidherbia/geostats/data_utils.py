import os
import numpy as np

########## Reminder : dataset structure ###########
# {datadir}/
#          |data/
#          |    |{year}/{date}/
#          |               ?  |shapefile/{parcel}/
#          |               ?  |                  |faidh/faidh.shp
#          |               ?  |                  |limites/{parcel}.shp
#          |                  |
#          |                  |raster/{parcel}.tif
#          |
#          |result/
#                 |{year}/
#          |      |      |{date}/
#          |      |      |      |{parcel}/{parcel}_tree_{i}_ms.tif
#          |      |      |   ?  |{parcel}/{parcel}_tree_{i}_mask_voronoi.tif
#          |      |      |   ?  |{parcel}/{parcel}_tree_{i}_mask_faidherbia.tif
#          |      |      |      |{parcel}/{parcel}_tree_{i}_plants.tif
#          |      |      |      
#          |      |      |yearly/{parcel}/
#          |      |                      |{parcel}_tree_{i}_biomass.tif
#          |      |                      |{parcel}_tree_{i}_yield.tif        
#          |      |                ?     |summary.csv   (Fields : Name, parcel, index, lat, lon, surface, radius, orientation)
#          |      |
#          |      |
#          |      |atlases/
#          |      |        |{atlas_name}/
#          |      |        |             |individuals_selected.csv
#          |      |        |             |ms_mean.tif
#          |      |        |             |ndvi_mean.tif
#          |      |        |             |ndvi_sigma.tif
#          |      |        |             |mean_fcover.tif
#          |      |        |             |mean_biomass.tif
#          |      |        |             |sigma_biomass.tif
#          |      |        |             |mean_yield.tif
#          |      |        |             |mean_biomass.tif
#          |      |
#          |transfer_files/...
#          |
# with :
# {datadir} the path to the main directory for accessing input data and results, result of calling get_main_directory()
# {year} the year of the aerial imaging, result of calling get_aerial_imaging_dates(), and getting the second argument
# {date} the date of the aerial imaging, result of calling get_aerial_imaging_dates(), and getting the first argument
# {parcel} the name of the parcel, result of calling get_parcel_list()
# {i} the index of the tree in the parcel, starting from 0
#
####################################################



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
    print(full_dates)
    #Copy the array dates, but with keeping only the years
    years=np.array([date[:4] for date in full_dates])
    return full_dates,years


def n_max_faidherbia():
    return 10000

def get_parcel_list(it_is_just_a_test=False):
    if(it_is_just_a_test):
        return np.array(["P01"])
    else:
        return np.array(["P01","P02","P04","P05","P08","P09","P10","P11"])

def get_ndvi_threshold_for_fcover():
    return 0.2

def create_result_dirs(datadir,years,dates,parcels):
    if not os.path.isdir(datadir+'result/'):
        os.mkdir(datadir+'result/')
    if not os.path.isdir(datadir+'transfer_files/'):
        os.mkdir(datadir+'transfer_files/')

    for i in range(years.size):
        resultdir=datadir+'result/atlases/'
        if not os.path.isdir(resultdir):
            os.mkdir(resultdir)
        year=years[i]
        date=dates[i]
        resultdir=datadir+'result/'+year+'/'
        if not os.path.isdir(resultdir):
            os.mkdir(resultdir)
        resultdir=datadir+'result/'+year+'/'+date+'/'
        if not os.path.isdir(resultdir):
            os.mkdir(resultdir)
        for p in parcels:
            resultdir=datadir+'result/'+year+'/'+date+'/'+p
            if not os.path.isdir(resultdir):
                os.mkdir(resultdir)


