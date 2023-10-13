import rasterio
import faidherbia.geostats.geometry_utils as geometry_utils
import faidherbia.geostats.data_utils as data_utils
from PIL import Image
import numpy as np
import rasterio.features
import rasterio.mask
import matplotlib.pyplot as plt

print("Testing geometry_utils.get_voronoi_regions...")


# Open the GeoTIFF raster file and get the corresponding GeoTransform
parcel="P01"
year="2021"
date="2021_08_05"
region_polys,region_pts,vect_centroids,parcel_shape=geometry_utils.get_voronoi_regions(parcel,year,date)
src= rasterio.open("/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/"+year+"/"+date+"/raster/"+parcel+".tif")
transform = src.transform
dimX=src.shape[0]
dimY=src.shape[1]


item=1
lon_lat_coord = vect_centroids[item]
polygon=region_polys[item]
row, col = rasterio.transform.rowcol(transform, lon_lat_coord.x, lon_lat_coord.y)
patch_size_in_pixels=4000
window_area = rasterio.windows.Window(col-patch_size_in_pixels/2, row-patch_size_in_pixels/2, patch_size_in_pixels, patch_size_in_pixels)
coords=list(polygon.exterior.coords)

def get_patch_of_raster_around_tree(src,patch_size_in_pixels,lon_lat_coord):
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

if(True):
    patch=get_patch_of_raster_around_tree(src,patch_size_in_pixels,lon_lat_coord)
else:
    #Extract a patch of the raster around the tree
    patch = src.read(window=window_area, masked=False)
    #Extract dimX and dimY of src

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

img=Image.fromarray(patch[0,:,:])    
img.save("/home/rfernandez/Bureau/test_ms.tif")    
imgms=patch[0,:,:]

def get_patch_of_polygon_mask_around_tree(src,patch_size_in_pixels,lon_lat_coord,polygon):
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

if(True):
    patch=get_patch_of_polygon_mask_around_tree(src,patch_size_in_pixels,lon_lat_coord,polygon)

else:
    mask_full_parcel = rasterio.features.geometry_mask([polygon], out_shape=src.shape, transform=src.transform, invert=True)
    print(np.shape(mask_full_parcel))
    ranges=window_area.toranges()
    min_x=(int)(ranges[0][0])
    min_y=(int)(ranges[1][0])
    max_x=(int)(ranges[0][1])
    max_y=(int)(ranges[1][1])
    print("Bbox voronoi X : [ "+str(min_x)+" , "+str((min_x+max_x)/2)+" , "+str(max_x)+" ]")
    print("Bbox voronoi Y : [ "+str(min_y)+" , "+str((min_y+max_y)/2)+" , "+str(max_y)+" ]")
    patch = mask_full_parcel[max(0,min_x):min(dimX,max_x),max(0,min_y):min(dimY,max_y)]
    print(np.shape(patch))
    if((row-patch_size_in_pixels/2)<0):
        add_rows=(int)(-row+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((add_rows,patch.shape[1])),patch),axis=0)
    print(np.shape(patch))
    if((row+patch_size_in_pixels/2)>dimX):
        add_rows=(int)(row+patch_size_in_pixels/2-dimX)
        patch=np.concatenate((patch,np.zeros((add_rows,patch.shape[1]))),axis=0)
    print(np.shape(patch))
    if((col-patch_size_in_pixels/2)<0):
        add_cols=(int)(-col+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((patch.shape[0],add_cols)),patch),axis=1)
    print(np.shape(patch))
    if((col+patch_size_in_pixels/2)>dimY):
        add_cols=(int)(col+patch_size_in_pixels/2-dimY)
        patch=np.concatenate((patch,np.zeros((patch.shape[0],add_cols))),axis=1)
    print(np.shape(patch))
img=Image.fromarray(patch[:,:])    
img.save("/home/rfernandez/Bureau/test_mask.tif")    
#--> Makes a patch with not the good size, but with the tree centered in 2000, 2000
#--> How the hell does this work ? Ranges does not now nothing
imgmask=patch[:,:]

ax1=plt.subplot(1,2,1)
ax1.imshow(imgms,vmin=0,vmax=40000)
ax2=plt.subplot(1,2,2)
ax2.imshow(imgmask,vmin=0,vmax=1)
plt.show()

#   print("Pixel coordinates row,col of patch center:")
#   print(row, col)

#Access a polygon and compute its centroid and the center of its bounding box
#for item in range(0,len(region_polys)):
#    coords=list(region_polys[item].exterior.coords)
#    min_x=min([coord[0] for coord in coords])
#    min_y=min([coord[1] for coord in coords])
#    max_x=max([coord[0] for coord in coords])
#    max_y=max([coord[1] for coord in coords])
    #Display min, avg and max along x in a single line
#    print("Bbox voronoi X : [ "+str(min_x)+" , "+str((min_x+max_x)/2)+" , "+str(max_x)+" ]")
#    print("Bbox voronoi Y : [ "+str(min_y)+" , "+str((min_y+max_y)/2)+" , "+str(max_y)+" ]")
#    print("Faidherbia centroid="+str(vect_centroids[item]))


    # Convert the latitude/longitude coordinate of the centroid to pixel coordinates and compute the windows for patching
#    window_area = rasterio.windows.Window(col-patch_size_in_pixels/2, row-patch_size_in_pixels/2, patch_size_in_pixels, patch_size_in_pixels)
 #   ranges=window_area.toranges()
  #  min_x=(int)(ranges[0][0])
   # min_y=(int)(ranges[1][0])
   # max_x=(int)(ranges[0][1])
   # max_y=(int)(ranges[1][1])
    #print("window_avg_x="+str((min_x+max_x)/2))
    #print("window_avg_y="+str((min_y+max_y)/2))


if(False):
    #Extract a patch of the raster around the tree
    patch = src.read(window=window_area, masked=False)
    #Extract dimX and dimY of src

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
    img=Image.fromarray(patch[0,:,:])    
    img.save("/home/rfernandez/Bureau/test_ms.tif")    
    imgms=patch[0,:,:]


    mask_full_parcel = rasterio.features.geometry_mask([polygon], out_shape=src.shape, transform=src.transform, invert=True)
    print(np.shape(mask_full_parcel))
    patch = mask_full_parcel[max(0,min_x):min(dimX,max_x),max(0,min_y):min(dimY,max_y)]
    print(np.shape(patch))
    if((row-patch_size_in_pixels/2)<0):
        add_rows=(int)(-row+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((add_rows,patch.shape[1])),patch),axis=0)
    print(np.shape(patch))
    if((row+patch_size_in_pixels/2)>dimX):
        add_rows=(int)(row+patch_size_in_pixels/2-dimX)
        patch=np.concatenate((patch,np.zeros((add_rows,patch.shape[1]))),axis=0)
    print(np.shape(patch))
    if((col-patch_size_in_pixels/2)<0):
        add_cols=(int)(-col+patch_size_in_pixels/2)
        patch=np.concatenate((np.zeros((patch.shape[0],add_cols)),patch),axis=1)
    print(np.shape(patch))
    if((col+patch_size_in_pixels/2)>dimY):
        add_cols=(int)(col+patch_size_in_pixels/2-dimY)
        patch=np.concatenate((patch,np.zeros((patch.shape[0],add_cols))),axis=1)
    print(np.shape(patch))
    img=Image.fromarray(patch[:,:])    
    img.save("/home/rfernandez/Bureau/test_mask.tif")    
    #--> Makes a patch with not the good size, but with the tree centered in 2000, 2000
    #--> How the hell does this work ? Ranges does not now nothing
    imgmask=patch[:,:]

    ax1=plt.subplot(1,2,1)
    ax1.imshow(imgms,vmin=0,vmax=40000)
    ax2=plt.subplot(1,2,2)
    ax2.imshow(imgmask,vmin=0,vmax=1)
    plt.show()



# Create the mask for the polygon in the full raster
mask_full_parcel = rasterio.features.geometry_mask([polygon], out_shape=src.shape, transform=src.transform, invert=False)
#show this mask and the original raster side to side on a two plot figure with plt.imshow
ax1=plt.subplot(1,2,1)
# write the 6-band geotiff to /home/rfernandez/Bureau/test_full_ms.tif
a=src.read()
from skimage import io
#with io, write a
io.imsave('/home/rfernandez/Bureau/test_full_ms.tif', a)

#io.write(a, '/home/rfernandez/Bureau/test_full_ms.tif', driver='GTiff')
#write

ax1.imshow(src.read(1), cmap='pink')
ax2=plt.subplot(1,2,2)
ax2.imshow(mask_full_parcel, cmap='pink')
plt.show()


#plt.imshow(mask_full_parcel)
#plt.show()

image_mask, out_transform = rasterio.mask.mask(src, shapes = [polygon], crop = False)
image_mask = np.where(image_mask < -1, 0, image_mask)
plt.imshow(image_mask[0,:,:])
plt.show()
Ms_img = rasterio.open('/home/rfernandez/Bureau/test_mask.tif','w',driver='Gtiff',
                        width=patch_size_in_pixels,
                        height=patch_size_in_pixels,
                        count=1,
                        crs=src.crs,
                        dtype='float32',
                        transform=src.transform,
                        compress='lzw')
print(np.shape(image_mask))


# reshape by forgetting the 0th dimension, then write
Ms_img.write(image_mask[0,:,:].reshape(image_mask.shape[1],image_mask.shape[2]),1)
Ms_img.close()



patch_ms_data= src.read(window=window_area, masked=True)
patch_mask= mask[min_x:max_x,min_y:max_y]

#print the shape of these two objects
print("patch_ms_data.shape="+str(patch_ms_data.shape))
print("patch_mask.shape="+str(patch_mask.shape))

#plot both on a single figure
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(patch_ms_data[0,:,:])
ax2.imshow(patch_mask[:,:])
plt.show()
