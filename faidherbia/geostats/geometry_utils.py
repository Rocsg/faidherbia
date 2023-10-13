"""
This module contains utility functions for working with geometry and spatial data.

Functions:
- correct_order(regions_polys, vect_centroids): Sorts a list of polygons based on their centroid location.
- get_voronoi_regions(parcel, year, date): Computes Voronoi regions for a given parcel, year and date.
- get_patch_of_raster_around_tree(src, patch_size_in_pixels, lon_lat_coord): Extracts a patch of a raster around a given tree, given its latitude/longitude coordinate.
- get_patch_of_polygon_mask_around_tree(src, patch_size_in_pixels, lon_lat_coord, polygon): Extracts a patch of a given size around a tree, defined by its centroid coordinates and a polygon. The resulting patch is a mask of the polygon.
"""
from shapely.geometry import Point
from geovoronoi import voronoi_regions_from_coords
import geopandas as gpd
import rasterio
import rasterio.features
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches

import faidherbia.geostats.data_utils as data_utils


#Activate this flag to run the tests
Testing=False



def correct_order(regions_polys, vect_centroids):
    """
    Sorts a list of polygons based on their centroid location.

    Args:
    regions_polys (list): A list of polygons to be sorted.
    vect_centroids (list): A list of centroids used to sort the polygons.

    Returns:
    list: A new list of polygons sorted based on their centroid location.
    """
    #Build a new list of polygons
    new_list=[]
    for i in range(0,len(vect_centroids)):
        #Find the index of the polygon that contains the centroid
        for j in range(0,len(regions_polys)):
            if(regions_polys[j].contains(vect_centroids[i])):
                new_list.append(regions_polys[j])
                break
    return new_list
    




def get_faidherbia_crown_and_voronoi_regions(parcel, year, date):
    """
    Computes Voronoi regions for a given parcel, year and date.

    Args:
    - parcel (str): name of the parcel
    - year (str): year of the data
    - date (str): date of the data

    Returns:
    - regions_polys (list): list of polygons representing the Voronoi regions
    - region_pts (list): list of points representing the Voronoi regions
    - vect_centroids (list): list of centroids of the faidherbia polygons
    - parcel_shape (Polygon): polygon representing the shape of the parcel
    """
    datadir = data_utils.get_main_directory()
    faidherbia_shapes = gpd.read_file(datadir + "data/" + year + "/" + date + "/shapefile/" + parcel + "/faidh/faidh.shp")
    parcel_shape = gpd.read_file(datadir + "data/" + year + "/" + date + "/shapefile/" + parcel + "/limites/" + parcel + ".shp").iloc[0].geometry
    faidherbia_polygons = faidherbia_shapes['geometry']
    vect_centroids = [faidherbia_polygon.centroid for faidherbia_polygon in faidherbia_polygons]
    regions_polys, region_pts = voronoi_regions_from_coords(vect_centroids, parcel_shape, per_geom=False)
    print("parcshape" + str(parcel_shape.centroid))
    regions_polys = correct_order(regions_polys, vect_centroids)

    return regions_polys, region_pts, vect_centroids, parcel_shape,faidherbia_polygons







def get_patch_of_raster_around_tree(src, patch_size_in_pixels, lon_lat_coord):
    """
    Extracts a patch of a raster around a given tree, given its latitude/longitude coordinate.

    Args:
        src (rasterio.DatasetReader): The raster dataset to extract the patch from.
        patch_size_in_pixels (int): The size of the patch to extract, in pixels.
        lon_lat_coord (shapely.geometry.Point): The latitude/longitude coordinate of the tree.

    Returns:
        numpy.ndarray: The extracted patch of the raster around the tree.
    """
    # Convert the latitude/longitude coordinate of the centroid to pixel coordinates and compute the windows for patching
    row, col = rasterio.transform.rowcol(src.transform, lon_lat_coord.x, lon_lat_coord.y)
    window_area = rasterio.windows.Window(col-patch_size_in_pixels/2, row-patch_size_in_pixels/2, patch_size_in_pixels, patch_size_in_pixels)
    #Extract a patch of the raster around the tree
    patch = src.read(window=window_area, masked=False)
    #Extract dimX and dimY of src
    dimX=src.shape[0]
    dimY=src.shape[1]
    #Add rows and columns to go to the expected patch size
    if((row-patch_size_in_pixels/2)<0):
        add_rows=(int)(-row+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((patch.shape[0],add_rows,patch.shape[2])),patch),axis=1)
    if((row+patch_size_in_pixels/2)>dimX):
        add_rows=(int)(row+patch_size_in_pixels/2-dimX)
        patch=np.concatenate((patch,np.zeros((patch.shape[0],add_rows,patch.shape[2]))),axis=1)
    if((col-patch_size_in_pixels/2)<0):
        add_cols=(int)(-col+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((patch.shape[0],patch.shape[1],add_cols)),patch),axis=2)
    if((col+patch_size_in_pixels/2)>dimY):
        add_cols=(int)(col+patch_size_in_pixels/2-dimY)
        patch=np.concatenate((patch,np.zeros((patch.shape[0],patch.shape[1],add_cols))),axis=2)
    return patch





def get_patch_of_polygon_mask_around_tree(src, patch_size_in_pixels, lon_lat_coord, polygon):
    """
    Extracts a patch of a given size around a tree, defined by its centroid coordinates and a polygon. The resulting patch is a mask of the polygon.

    Args:
        src (rasterio.DatasetReader): The raster dataset to extract the patch from.
        patch_size_in_pixels (int): The size of the patch to extract, in pixels.
        lon_lat_coord (shapely.geometry.Point): The centroid coordinates of the tree, in longitude and latitude.
        polygon (shapely.geometry.Polygon): The polygon defining the tree.

    Returns:
        numpy.ndarray: The extracted patch of the given size around the tree.
    """
    # Convert the latitude/longitude coordinate of the centroid to pixel coordinates and compute the windows for patching
    row, col = rasterio.transform.rowcol(src.transform, lon_lat_coord.x, lon_lat_coord.y)
    window_area = rasterio.windows.Window(col-patch_size_in_pixels/2, row-patch_size_in_pixels/2, patch_size_in_pixels, patch_size_in_pixels)
    #Extract dimX and dimY of src
    dimX=src.shape[0]
    dimY=src.shape[1]
    #Add rows and columns to go to the expected patch size
    ranges=window_area.toranges()
    min_x=(int)(ranges[0][0])
    min_y=(int)(ranges[1][0])
    max_x=(int)(ranges[0][1])
    max_y=(int)(ranges[1][1])
    #Create the mask for the polygon in the full raster and extract the patch in the polygon bbox
    mask_full_parcel = rasterio.features.geometry_mask([polygon], out_shape=src.shape, transform=src.transform, invert=True)    
    patch = mask_full_parcel[max(0,min_x):min(dimX,max_x),max(0,min_y):min(dimY,max_y)]
    #Add missing rows and columns to get to the expected patch size
    if((row-patch_size_in_pixels/2)<0):
        add_rows=(int)(-row+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((add_rows,patch.shape[1])),patch),axis=0)
    if((row+patch_size_in_pixels/2)>dimX):
        add_rows=(int)(row+patch_size_in_pixels/2-dimX)
        patch=np.concatenate((patch,np.zeros((add_rows,patch.shape[1]))),axis=0)
    if((col-patch_size_in_pixels/2)<0):
        add_cols=(int)(-col+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((patch.shape[0],add_cols)),patch),axis=1)
    if((col+patch_size_in_pixels/2)>dimY):
        add_cols=(int)(col+patch_size_in_pixels/2-dimY)
        patch=np.concatenate((patch,np.zeros((patch.shape[0],add_cols))),axis=1)
    return patch






def draw_parcel_polygon_and_centroid(parcel_poly,region_polys,vect_centroids,item):
    """
    Draws a plot of the parcel contour and the voronoi polygons, highlighting a given faidherbia tree.

    Args:
    - parcel_poly: a shapely Polygon object representing the parcel polygon
    - region_polys: a list of shapely Polygon objects representing the region polygons
    - vect_centroids: a list of shapely Point objects representing the centroids of the region polygons
    - item: an integer representing the item to be plotted

    Returns:
    - None
    """
    #Create a list of future coordinates and values. N lines, 3 columns
    #The first column is the X coordinate, the second is the Y coordinate, the third is the value
    data=np.zeros((1000,4))
    lines=np.zeros((1000,8))
    incr=0
    value=0
    fig, ax = plt.subplots()
    to_test=[2]
    ax.add_patch(matplotlib.patches.Polygon(parcel_poly.exterior, color='black', alpha=0.5,closed=True,fill=False))

    #Add coordinates of region_poly to data and trees
    for i in range(0,len(region_polys)):
        value=value+(11*i)%7+2
        si=40
        coords=list(region_polys[i].exterior.coords)
        #If i is in to_test
        if(i in to_test):
            for coord in coords:
                data[incr]=[xnorm(coord[0]),ynorm(coord[1]),value,si]
                incr=incr+1
        ax.add_patch(matplotlib.patches.Polygon(coords, color='black', alpha=0.5,closed=True,fill=False))
        coords=list(vect_centroids[i].coords)            
        si=150
        for coord in coords:
            data[incr]=[xnorm(coord[0]),ynorm(coord[1]),value,si]
            incr=incr+1

    #Plot the data
    ax.scatter(data[:,0], data[:,1], c=data[:,2],cmap='turbo', s=data[:,3],edgecolor='black')
    ax.set_xlim([-16.452630, -16.451727])
    ax.set_ylim([14.494248,14.495331])
    plt.show()





# Minor helpers to express coordinates in a more handsome way, for printing
def xnorm(x):
    return x
 #   return (x+16.4521025)*100000

def ynorm(y):
    return y
    #  return (y-14.494859)*100000

def pointnorm(point):
    p = Point(xnorm(point.x), ynorm(point.y))
    return p




#Unit test routine. Test all the functions of the module
if(Testing):
    #Extract basic data from the dataset
    print("Testing geometry_utils.get_voronoi_regions...")
    parcel="P01"
    year="2021"
    date="2021_08_05"
    region_polys,region_pts,vect_centroids,parcel_shape,faidherbia_crowns=get_faidherbia_crown_and_voronoi_regions(parcel,year,date)
    print("-----Region_polys:")
    print(region_polys)

    #Select a faidherbia, compute some stats about its shape, its polygon' shape, and its centroid
    item=2
    coords=list(region_polys[item].exterior.coords)
    centr_norm=pointnorm(vect_centroids[item])
    min_x=xnorm(min([coord[0] for coord in coords]))
    min_y=ynorm(min([coord[1] for coord in coords]))
    max_x=xnorm(max([coord[0] for coord in coords]))
    max_y=ynorm(max([coord[1] for coord in coords]))
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
    draw_parcel_polygon_and_centroid(parcel_shape,region_polys,vect_centroids,item)

    #Extract and view a patch of this tree
    src= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/"+parcel+".tif")
    patch_size_in_pixels=4000

    patch=get_patch_of_raster_around_tree(src,patch_size_in_pixels,vect_centroids[item])
    imggreenband=patch[1,:,:]

    patch=get_patch_of_polygon_mask_around_tree(src,patch_size_in_pixels,vect_centroids[item],region_polys[item])
    imgmask=patch[:,:]

    ax1=plt.subplot(1,2,1)
    ax1.imshow(imggreenband,vmin=0,vmax=40000)
    ax2=plt.subplot(1,2,2)
    ax2.imshow(imgmask,vmin=0,vmax=1)
    plt.show()

    print("Test passed !")
