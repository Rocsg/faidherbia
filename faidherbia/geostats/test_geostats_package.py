import unittest
import rasterio
import matplotlib.pyplot as plt
import faidherbia.geostats.data_utils as data_utils
import faidherbia.geostats.geometry_utils as geometry_utils


#Run it in command line using:
# python -m unittest test_geostats_package.py

class TestGeostatsPackage(unittest.TestCase):
    def test_data_utils(self):
        print("Testing data_utils.is_mansour...")
        print("is_mansour()="+str(data_utils.is_mansour()))
        print("Test passed !")


        print("Testing data_utils.get_main_directory...")
        print("get_main_directory()="+str(data_utils.get_main_directory()))
        print("Test passed !")

        print("Testing data_utils.get_aerial_imaging_dates...")
        print("get_aerial_imaging_dates()="+str(data_utils.get_aerial_imaging_dates(it_is_just_a_test=True)))
        print("Test passed !")


        print("Testing data_utils.get_parcel_list...")
        print("get_parcel_list()="+str(data_utils.get_parcel_list(it_is_just_a_test=True)))
        print("Test passed !")


        print("Testing data_utils.create_result_dirs...")
        datadir=data_utils.get_main_directory()
        years,dates=data_utils.get_aerial_imaging_dates(it_is_just_a_test=True)
        parcels=data_utils.get_parcel_list(it_is_just_a_test=True)
        data_utils.create_result_dirs(datadir,years,dates,parcels)
        print("Test passed !")

    def test_geometry_utils():
        #Extract basic data from the dataset
        print("Testing geometry_utils.get_voronoi_regions...")
        parcel="P01"
        year="2021"
        date="2021_08_05"
        region_polys,region_pts,vect_centroids,parcel_shape,faidherbia_crowns=geometry_utils.get_faidherbia_crown_and_voronoi_regions(parcel,year,date)
        print("-----Region_polys:")
        print(region_polys)

        #Select a faidherbia, compute some stats about its shape, its polygon' shape, and its centroid
        item=2
        coords=list(region_polys[item].exterior.coords)
        centr_norm=geometry_utils.pointnorm(vect_centroids[item])
        min_x=geometry_utils.xnorm(min([coord[0] for coord in coords]))
        min_y=geometry_utils.ynorm(min([coord[1] for coord in coords]))
        max_x=geometry_utils.xnorm(max([coord[0] for coord in coords]))
        max_y=geometry_utils.ynorm(max([coord[1] for coord in coords]))
        print("Bbox voronoi X : [ "+str(min_x)+" , "+str((min_x+max_x)/2)+" , "+str(max_x)+" ]")
        print("Bbox voronoi Y : [ "+str(min_y)+" , "+str((min_y+max_y)/2)+" , "+str(max_y)+" ]")
        print("Faidherbia centroid="+str(centr_norm))
        diff_x=(centr_norm.x-(min_x+max_x)/2)
        diff_y=(centr_norm.y-(min_y+max_y)/2)
        print("diff_x meters="+str(diff_x))
        print("diff_y_meters="+str(diff_y))
        print("Is the centroid inside the polygon ?")
        print(region_polys[item].contains(vect_centroids[item]))

        #Draw the parcel, and highlight a faidherbia
        geometry_utils.draw_parcel_polygon_and_centroid(parcel_shape,region_polys,vect_centroids,item)

        #Extract and view a patch of this tree
        src= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/"+parcel+".tif")
        patch_size_in_pixels=4000

        patch=geometry_utils.get_patch_of_raster_around_tree(src,patch_size_in_pixels,vect_centroids[item])
        imggreenband=patch[1,:,:]

        patch=geometry_utils.get_patch_of_polygon_mask_around_tree(src,patch_size_in_pixels,vect_centroids[item],region_polys[item])
        imgmask=patch[:,:]

        ax1=plt.subplot(1,2,1)
        ax1.imshow(imggreenband,vmin=0,vmax=40000)
        ax2=plt.subplot(1,2,2)
        ax2.imshow(imgmask,vmin=0,vmax=1)
        plt.show()

        print("Test passed !")
