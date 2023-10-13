import os
import rasterio
import skimage.io
import faidherbia.geostats.geometry_utils as geometry_utils
import faidherbia.geostats.data_utils as data_utils
import numpy as np
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
def make_inventory_of_faidherbia(it_is_just_a_test=False):
    """
    Extracts faidherbia inventory data from aerial imaging data and saves it in a CSV file.
    The inventory contains one line per faidherbia, storing the following information:
    - Parcel name
    - Faidherbia index
    - Radius of the faidherbia crown (in meters)
    - Perimeter of the faidherbia crown (in meters)
    - Axis of the faidherbia crown (in degrees)
    - X coordinate of the faidherbia centroid (in meters)
    - Y coordinate of the faidherbia centroid (in meters)

    Args:
    it_is_just_a_test (bool): If True, uses test data. If False, uses actual data.

    Returns:
    None
    """
    datadir=data_utils.get_main_directory()
    parcels=data_utils.get_parcel_list(it_is_just_a_test)
    dates,years=data_utils.get_aerial_imaging_dates(it_is_just_a_test)
    data_utils.create_result_dirs(datadir,years,dates,parcels)

    #Create a numpy array that will be incremented for each faidherbia
    faidherbia_inventory=[]
    faidh_info=["Parcel","Index","Radius","Perimeter","Axis","CenterX","CenterY","Date","Year","Selected"]
    faidherbia_inventory.append(faidh_info) 
    for indparcel in range(0,len(parcels)):
        parcel=parcels[indparcel]
        for indyear in range(0,len(years)):
            year=years[indyear]
            date=dates[indyear]
            region_polys,region_pts,vect_centroids,parcel_shape,faidherbia_crowns=geometry_utils.get_faidherbia_crown_and_voronoi_regions(parcel,year,date)
            for indfaid in range(0,len(region_polys)):
                faidherbia_crown=faidherbia_crowns[indfaid]
                faidherbia_centroid=vect_centroids[indfaid]
                axis=90
                perimeter=faidherbia_crown.length
                radius=np.sqrt(faidherbia_crown.area/(np.pi))
                faidh_info=[parcel,indfaid,radius*111000,perimeter*111000,axis,faidherbia_centroid.x,faidherbia_centroid.y,date,year,0]

                faidherbia_inventory.append(faidh_info) 
    print(faidherbia_inventory)

    #Save the inventory in a CSV file
    np.savetxt(datadir+'result/faidherbia_inventory.csv', faidherbia_inventory, delimiter=',', fmt='%s')
    return pd.read_csv(datadir+'result/faidherbia_inventory.csv')



def select_faidherbia(df,size='all',date='2021_08_05'):
    df.loc['Selected'] = 0

    #Select according to date
    df.loc[df['Date'] == date, 'Selected'] = 1
    
    #Select according to size
    sorted_df=df.sort_values(by=['Radius'])
    one_third_radius=sorted_df.iloc[int(sorted_df.shape[0]/3)]['Radius']
    two_third_radius=sorted_df.iloc[int((sorted_df.shape[0]*2)/3)]['Radius']
    if(size=='small'):
        df.loc[df['Radius'] > one_third_radius, 'Selected'] = 0

    if(size=='large'):
        df.loc[df['Radius'] < two_third_radius, 'Selected'] = 0

    if(size=='medium'):
        df.loc[df['Radius'] >= two_third_radius, 'Selected'] = 0
        df.loc[df['Radius'] <= one_third_radius, 'Selected'] = 0
    return df





def batch_extraction_with_no_save_and_atlas_building(it_is_just_a_test=False,patch_size_in_pixels=4000,size='all',date='2021_08_05'):
    ndvi_threshold_for_fcover=data_utils.get_ndvi_threshold_for_fcover()
    datadir=data_utils.get_main_directory()
    parcels=data_utils.get_parcel_list(it_is_just_a_test)
    dates,years=data_utils.get_aerial_imaging_dates(it_is_just_a_test)

    exp_name=date+'_'+size
    data_utils.create_result_dirs(datadir,years,dates,parcels)
    expdatadir=datadir+'result/atlases/'+exp_name
    if not os.path.isdir(expdatadir):
        os.mkdir(expdatadir)

    inventory=make_inventory_of_faidherbia(it_is_just_a_test)
    inventory=select_faidherbia(inventory,size,date)
    #write inventory as a csv
    inventory.to_csv(expdatadir+'/inventory.csv', index=False)



    weight=np.zeros((patch_size_in_pixels,patch_size_in_pixels))
    ms=np.zeros((6,patch_size_in_pixels,patch_size_in_pixels))
    ndvi=np.zeros((patch_size_in_pixels,patch_size_in_pixels))
    fcover=np.zeros((patch_size_in_pixels,patch_size_in_pixels))
    faid_biomass=np.zeros((patch_size_in_pixels,patch_size_in_pixels))
    faid_yield=np.zeros((patch_size_in_pixels,patch_size_in_pixels))
    for indparcel in range(0,len(parcels)):
        parcel=parcels[indparcel]
        print("-----> Processing parcel "+parcel)
        for indyear in range(0,len(years)):
            year=years[indyear]
            date=dates[indyear]
            print("-> Processing year "+year+" and date "+date)
            #Import the geometry of the parcel, faidherbia, compute and import voronoi regions geometry
            region_polys,region_pts,vect_centroids,parcel_shape,faidherbia_crowns=geometry_utils.get_faidherbia_crown_and_voronoi_regions(parcel,year,date)
            #If none of the faidherbia for this parcel and this date are selected, skip the parcel
            if(inventory.loc[(inventory['Parcel'] == parcel) & (inventory['Date'] == date) & (inventory['Selected'] == 1)].shape[0]==0):
                continue
            #Import the raster of the parcel
            print("loading raster", end = '')
            parcel_raster= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/"+parcel+".tif")
            print("...ok")

            # For each faidherbia of this parcel/date
            for indfaid in range(0,len(region_polys)):
                print("-----> Processing faidherbia "+str(indfaid))
                #if this faidherbia is not selected with this date and this parcel, skip
                if(inventory.loc[(inventory['Parcel'] == parcel) & (inventory['Date'] == date) & (inventory['Index'] == indfaid) & (inventory['Selected'] == 1)].shape[0]==0):
                    continue
                faidherbia_voronoi=region_polys[indfaid]
                faidherbia_crown=faidherbia_crowns[indfaid]
                faidherbia_centroid=vect_centroids[indfaid]

                #Extract the mask of the faidherbia crown and save it in result
                faidherbia_crown_mask=geometry_utils.get_patch_of_polygon_mask_around_tree(parcel_raster, patch_size_in_pixels, faidherbia_centroid, faidherbia_crown)
                #Extract the mask of the faidherbia voronoi region
                faidherbia_voronoi_mask=geometry_utils.get_patch_of_polygon_mask_around_tree(parcel_raster, patch_size_in_pixels, faidherbia_centroid, faidherbia_voronoi)
                #Extract the patch of the parcel as a 6 bands image
                faidherbia_ms_patch=geometry_utils.get_patch_of_raster_around_tree(parcel_raster, patch_size_in_pixels, faidherbia_centroid)
                faidherbia_ms_patch[faidherbia_ms_patch>65000]=0
                #Extract the patch of estimated biomass
#                faidherbia_biomass_patch=geometry_utils.get_patch_of_raster_around_tree(biomass_raster, patch_size_in_pixels, faidherbia_centroid)
                #compute mean of ms_patch along first dimension
                faidherbia_biomass_patch=np.mean(faidherbia_ms_patch,axis=0)
                #Extract the patch of estimated yield
#                faidherbia_yield_patch=geometry_utils.get_patch_of_raster_around_tree(yield_raster, patch_size_in_pixels, faidherbia_centroid)
                faidherbia_yield_patch=np.std(faidherbia_ms_patch,axis=0)
                #combine the masks by making a new one selecting only pixels not in crown mask but in voronoi mask
                faidherbia_voronoi_mask=faidherbia_voronoi_mask.astype(bool)
                faidherbia_crown_mask=faidherbia_crown_mask.astype(bool)
                faidherbia_glob_mask=np.logical_and(faidherbia_voronoi_mask,np.logical_not(faidherbia_crown_mask))

                #Add 1 in weight for each pixel in the global mask
                weight=weight+faidherbia_glob_mask
                debug=False

                if(debug):
                    print("weight")
                    plt.imshow(weight)
                    plt.show()

                #remove the pixels in the global mask from the ms patch
                for i in range(0,6):
                    msi=faidherbia_ms_patch[i]*faidherbia_glob_mask
                    ms[i]=ms[i]+msi
                    if(debug):
                        print("ms")
                        plt.imshow(ms[i])
                        plt.show()

                #Add the ndvi values in the global mask to the ndvi patch
                #First compute the ndvi by taking (ms[4]-ms[2])/(ms[4]+ms[2]), but 0 when ms[4]+ms[2]=0
                temp_ndvi=np.divide(np.subtract(ms[4],ms[2]),np.add(ms[4],ms[2]),out=np.zeros_like(ms[4]),where=np.add(ms[4],ms[2])!=0)*faidherbia_glob_mask
                ndvi=ndvi+temp_ndvi
                if(debug):
                    print("ndvi")
                    plt.imshow(ndvi)
                    plt.show()


                #Add the fcover values in the global mask to the fcover patch. Fcover is defined when ndvi>0.2
                #compute fcover as 1 when ndvi>0.2, 0 else
                temp_ndvi[temp_ndvi<0.2]=0
                temp_ndvi[temp_ndvi>=0.2]=1
                temp_fcover=temp_ndvi*faidherbia_glob_mask
                fcover=fcover+temp_fcover
                if(debug):
                    print("fcover")
                    plt.imshow(fcover)
                    plt.show()


                faid_biomass=faid_biomass+faidherbia_biomass_patch*faidherbia_glob_mask
                if(debug):
                    print("biomass")
                    plt.imshow(faid_biomass)
                    plt.show()

                faid_yield=faid_yield+faidherbia_yield_patch*faidherbia_glob_mask
                if(debug):
                    print("yield")
                    plt.imshow(faid_yield)
                    plt.show()



    #Set 1 to every 0 value in weight
    weight[weight==0]=1

    #Compute the mean
    for i in range(0,6):
        ms[i]=ms[i]/weight
        
    ndvi=ndvi/weight
    fcover=fcover/weight
    faid_biomass=faid_biomass/weight
    faid_yield=faid_yield/weight
    skimage.io.imsave(expdatadir+'/weight.tif', weight)
    skimage.io.imsave(expdatadir+'/ms.tif', ms)
    skimage.io.imsave(expdatadir+'/ndvi.tif', ndvi)
    skimage.io.imsave(expdatadir+'/fcover.tif', fcover)
    skimage.io.imsave(expdatadir+'/biomass.tif', faid_biomass)
    skimage.io.imsave(expdatadir+'/yield.tif', faid_yield)







def extract_and_save_patches_and_mask(it_is_just_a_test=False,patch_size_in_pixels=4000):
    """
    Extracts patches of a given size around faidherbia trees in a set of parcels, and saves them as images.
    Also saves the masks of the faidherbia crowns and voronoi regions as images.
    
    Args:
    - it_is_just_a_test (bool): whether to run the function in test mode (default: False)
    - patch_size_in_pixels (int): size of the patches to extract, in pixels (default: 4000)
    """
    # Import the data list
    it_is_just_a_test=True
    patch_size_in_pixels=4000
    datadir=data_utils.get_main_directory()
    parcels=data_utils.get_parcel_list(it_is_just_a_test)
    dates,years=data_utils.get_aerial_imaging_dates(it_is_just_a_test)


    # If necessary, create {datadir}/result/{year}/{date}/, for each year and date
    data_utils.create_result_dirs(datadir,years,dates,parcels)


    # For the given year / dates
    for indyear in range(0,len(years)):
        year=years[indyear]
        date=dates[indyear]
        print("-> Processing year "+year+" and date "+date)

        # For the given parcels
        for indparcel in range(0,len(parcels)):
            parcel=parcels[indparcel]
            print("-----> Processing parcel "+parcel)

            #Import the geometry of the parcel, faidherbia, compute and import voronoi regions geometry
            region_polys,region_pts,vect_centroids,parcel_shape,faidherbia_crowns=geometry_utils.get_faidherbia_crown_and_voronoi_regions(parcel,year,date)

            #Import the raster of the parcel
            parcel_raster= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/"+parcel+".tif")

            # For each faidherbia
            for indfaid in range(0,len(region_polys)):
                faidherbia_voronoi=region_polys[indfaid]
                faidherbia_crown=faidherbia_crowns[indfaid]
                faidherbia_centroid=vect_centroids[indfaid]

                #Extract the mask of the faidherbia crown and save it in result
                faidherbia_crown_mask=geometry_utils.get_patch_of_polygon_mask_around_tree(
                    parcel_raster, patch_size_in_pixels, faidherbia_centroid, faidherbia_crown)
                skimage.io.imsave(datadir+'result/'+year+'/'+date+'/'+parcel+'/faidherbia_'+str(indfaid)+'_crown_mask.tif', faidherbia_crown_mask)

                #Extract the mask of the faidherbia voronoi region
                faidherbia_voronoi_mask=geometry_utils.get_patch_of_polygon_mask_around_tree(
                    parcel_raster, patch_size_in_pixels, faidherbia_centroid, faidherbia_voronoi)
                skimage.io.imsave(datadir+'result/'+year+'/'+date+'/'+parcel+'/faidherbia_'+str(indfaid)+'_voronoi_mask.tif', faidherbia_voronoi_mask)

                #Extract the patch of the parcel as a 6 bands image
                faidherbia_ms_patch=geometry_utils.get_patch_of_raster_around_tree(parcel_raster, patch_size_in_pixels, faidherbia_centroid)
                skimage.io.imsave(datadir+'result/'+year+'/'+date+'/'+parcel+'/faidherbia_'+str(indfaid)+'_ms_patch.tif', faidherbia_ms_patch)

                #Get data for CSV (work in progress)

def run():
     batch_extraction_with_no_save_and_atlas_building(it_is_just_a_test=True,patch_size_in_pixels=4000,size='all',date='2021_08_05')   

run()