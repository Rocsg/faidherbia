from faidherbia.geostats.polar_viewer import polar_viewer_on_actual_data
import numpy as np
from skimage import io





#To make the figure of yield (result 6)
if(False):
    expdatadir='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/result/atlases/2021_08_05_all'
    mask_representative=np.load(expdatadir+'/mask_representative.npy')
    mapyield=io.imread(expdatadir+'/yield.tif')[0,:,:]
    min=81
    max=86
    mapyield[mask_representative[:,:]<1]=min
    polar_viewer_on_actual_data(mapyield,mask_representative,'/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Figures/result_6_yield.png',valmin=min,valmax=max,colormap='viridis')


#To make the figure of biomass (result 5)
if(False):
    expdatadir='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/result/atlases/2021_08_05_all'
    mask_representative=np.load(expdatadir+'/mask_representative.npy')
    biomass=io.imread(expdatadir+'/biomass.tif')[0,:,:]
    min=179
    max=205
    biomass[mask_representative[:,:]<1]=min
    polar_viewer_on_actual_data(biomass,mask_representative,'/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Figures/result_5_biomass.png',valmin=min,valmax=max,colormap='viridis')


#To make the figure of fcover (result 4)
if(True):
    expdatadir='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/result/atlases/2021_08_05_all'
    mask_representative=np.load(expdatadir+'/mask_representative.npy')
    fcover=np.load(expdatadir+'/fcover.npy')
    min=0.20
    max=0.32
    fcover[mask_representative[:,:]<1]=min
    polar_viewer_on_actual_data(fcover,mask_representative,'/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Figures/result_4_fcover.png',valmin=min,valmax=max,colormap='viridis')

#To make the figure of ndviplant (result 3)
if(True):
    expdatadir='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/result/atlases/2021_08_05_all'
    mask_representative=np.load(expdatadir+'/mask_representative.npy')
    ndviplant=io.imread(expdatadir+'/ndviplant.tif')
    min=0.32
    max=0.48
    ndviplant[mask_representative[:,:]<1]=min
    polar_viewer_on_actual_data(ndviplant,mask_representative,'/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Figures/result_3_ndviplant.png',valmin=min,valmax=max,colormap='viridis')


#To make the figure of ndvi (result 2)
if(False):
    expdatadir='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/result/atlases/2021_08_05_all'
    mask_representative=np.load(expdatadir+'/mask_representative.npy')
    ndvi=np.load(expdatadir+'/ndvi.npy')
    min=-0.05
    max=0.1
    ndvi[mask_representative[:,:]<1]=min
    polar_viewer_on_actual_data(ndvi,mask_representative,'/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Figures/result_2_ndvi.png',valmin=min,valmax=max,colormap='viridis')

#To make the figure of ground (result 1)
if(True):
    #import matplotlib.pyplot as plt
    #mask_representative=np.load(expdatadir+'/mask_representative.npy')
    #fig, (ax1, ax2) = plt.subplots(ncols=2)
    #ax1.imshow(mask_representative,cmap='gray')
    #ax2.imshow(mask_representative)
    #plt.show()
    #fig.savefig('/home/rfernandez/rgbgroundr.png',dpi=300)



    #To make... Idk
    """
    max_16_bit_value=65535
    rgbgroundr=np.load(expdatadir+'/rgbgroundr.npy')
    rgbgroundb=np.load(expdatadir+'/rgbgroundb.npy')
    rgbgroundg=np.load(expdatadir+'/rgbgroundg.npy')
    polar_viewer_on_actual_data(rgbgroundr,mask_representative,'/home/rfernandez/rgbgroundr.png',valmin=0,valmax=max_16_bit_value,colormap='gray')
    polar_viewer_on_actual_data(rgbgroundb,mask_representative,'/home/rfernandez/rgbgroundb.png',valmin=0,valmax=max_16_bit_value,colormap='gray')
    polar_viewer_on_actual_data(rgbgroundg,mask_representative,'/home/rfernandez/rgbgroundg.png',valmin=0,valmax=max_16_bit_value,colormap='gray')
    """


    #To make Idk
    """
    #Load image /home/rfernandez/TestRGB.tif, which is a 3 channel RGB image using scikit-image.io
    from skimage import io
    rgb=io.imread('/home/rfernandez/Bureau/TestRGB.tif')

    #Upsample of a factor 4
    from skimage.transform import rescale

    #split the three channels into three numpy arrau and compute polar_viewer on them
    rgbgroundr=rgb[:,:,0]
    rgbgroundg=rgb[:,:,1]
    rgbgroundb=rgb[:,:,2]

    rgbgroundr=rescale(rgbgroundr,4,anti_aliasing=True)
    rgbgroundg=rescale(rgbgroundg,4,anti_aliasing=True)
    rgbgroundb=rescale(rgbgroundb,4,anti_aliasing=True)
    max_8_bit_value=1
    polar_viewer_on_actual_data(rgbgroundr,mask_representative, '/home/rfernandez/Bureau/rgbgroundr.png',valmin=0,valmax=max_8_bit_value,colormap='gray')
    polar_viewer_on_actual_data(rgbgroundg,mask_representative,'/home/rfernandez/Bureau/rgbgroundg.png',valmin=0,valmax=max_8_bit_value,colormap='gray')
    polar_viewer_on_actual_data(rgbgroundb,mask_representative,'/home/rfernandez/Bureau/rgbgroundb.png',valmin=0,valmax=max_8_bit_value,colormap='gray')

    """
