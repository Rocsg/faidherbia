from faidherbia.geostats.polar_viewer import polar_viewer_on_actual_data
import numpy as np


expdatadir='/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Atlas/result/atlases/2021_08_05_all'

fcover=np.load(expdatadir+'/fcover.npy')
import matplotlib.pyplot as plt

mask_representative=np.load(expdatadir+'/mask_representative.npy')
rgbgroundr=np.load(expdatadir+'/rgbgroundr.npy')
rgbgroundb=np.load(expdatadir+'/rgbgroundb.npy')
rgbgroundg=np.load(expdatadir+'/rgbgroundg.npy')

max_16_bit_value=65535
polar_viewer_on_actual_data(rgbgroundr,mask_representative,'/home/rfernandez/rgbgroundr.png',valmin=0,valmax=max_16_bit_value,colormap='gray')
polar_viewer_on_actual_data(rgbgroundb,mask_representative,'/home/rfernandez/rgbgroundb.png',valmin=0,valmax=max_16_bit_value,colormap='gray')
polar_viewer_on_actual_data(rgbgroundg,mask_representative,'/home/rfernandez/rgbgroundg.png',valmin=0,valmax=max_16_bit_value,colormap='gray')

