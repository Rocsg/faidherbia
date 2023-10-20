
#Import a raster using rasterio
import rasterio
from rasterio.enums import Resampling
scaling_factor = 0.5
input_path='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/2021/2021_08_05/raster/P04.tif'
output_path='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/data/2021/2021_08_05/raster/P04_sub.tif'
dataset=rasterio.open(input_path)

#Subsample the raster by a factor 2
#Doc = https://rasterio.readthedocs.io/en/latest/topics/resampling.html
data = dataset.read(out_shape=(dataset.count,int(dataset.height * scaling_factor),int(dataset.width * scaling_factor)),resampling=Resampling.bilinear)
transform = dataset.transform * dataset.transform.scale((dataset.width / data.shape[-1]),(dataset.height / data.shape[-2]))
#write the raster
downsampledataset=rasterio.open(output_path,'w',driver='GTiff',width=int(dataset.width * scaling_factor),height=int(dataset.height * scaling_factor),count=dataset.count,dtype=data.dtype,crs=dataset.crs,transform=transform)
downsampledataset.write(data)

