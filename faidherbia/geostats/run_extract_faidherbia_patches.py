import os
import rasterio
import skimage.io
import faidherbia.geostats.geometry_utils as geometry_utils
import faidherbia.geostats.data_utils as data_utils

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
