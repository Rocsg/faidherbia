import numpy as np
import math
import matplotlib.pyplot as plt
import time
from skimage.transform import resize

#Create a copy of a numpy array, then replace each value to map the distance from the pixel to the target point (h0,w0) 
def get_radius_map(h0,w0,data):
    target_w,target_h=np.shape(data)
    rtab=np.array([[   ( (j-w0)*(j-w0)+(i-h0)*(i-h0)   )  for j in range (target_w) ] for i in range (target_h)]) 
    return np.sqrt(rtab)

#Create a copy of a numpy array, then replace each value to map the angles indicating the vector joining the target point (h0,w0) to the pixel 
def get_teta_map(h0,w0,data):
    target_w,target_h=np.shape(data)
    tetatab=np.array([[   ( math.atan2((h0-i),(j-w0))  )  for j in range (target_w) ] for i in range (target_h)]) 
    return tetatab


#This function transform a heatmap into a polar heatmap, with data averaged over quadrants
#h0, w0 are the coordinates of the center of the polar system in use
#rays is a vector of the successive ray in use : [ray_circle_in, ray_step_1,ray_step_2,....,ray_circle_out]
#tetas is a vector of the successive tetas in use : [teta_0, teta_1, teta_2]
# compute_polar_heatmap(data,)
def compute_polar_heatmap(data,h0,w0,rays,tetas,mask_studied_area,valmin):
    h=np.shape(data)[0]
    w=np.shape(data)[1]
    tetamap=get_teta_map(h0,w0,data)
    radiusmap=get_radius_map(h0,w0,data)
    n_rays=len(rays)
    n_tetas=len(tetas)
    result=np.zeros((len(rays),len(tetas)))
    polar_heatmap=np.zeros_like(data)

    #Compute the mean of control pixels
    #These are the pixels that are : radiusmap[:,:]>rays[n_rays-1], and mask_representative=1
    mask=radiusmap[:,:]>rays[n_rays-1]
    mask[mask_studied_area<1]=0
    valcontrol=np.mean(data[mask])
    print("valcontrol="+str(valcontrol))


    result[n_rays-1,:]=valcontrol
    mask=radiusmap[:,:]>rays[n_rays-1]
    polar_heatmap=polar_heatmap+mask*valcontrol

    #Set the value to zero inside the smaller circle, indicating the out-of-study area (where the tree is, for example)
#    mask=radiusmap[:,:]<rays[0]
    val_out=0
    result[0,:len(tetas)]=val_out
#    polar_heatmap=polar_heatmap+mask*val_out

    #Compute the average value for each quadrant in the area under study
    mask_rep=(mask_studied_area[:,:]>=1)
    for nr in range(n_rays-1):
        r1=rays[nr]
        r2=rays[nr+1]
        #These are the points lying in this ray interval
        mask_r=(radiusmap[:,:]>=r1) & (radiusmap[:,:]<r2)

        for nt in range(n_tetas):
            t1=tetas[nt]
            t2=1000
            if (nt!=(n_tetas-1)):
                t2=tetas[nt+1]
            #These are the points lying in this teta interval
            mask_t=(tetamap[:,:]>=t1) & (tetamap[:,:]<t2)

            #Combine ray interval and teta interval, and compute the average over these points
            mask=mask_r*mask_t*mask_rep

            val_out=np.nanmean(data[mask])
            result[nr+1,nt]=val_out
            polar_heatmap=polar_heatmap+mask*val_out

    #Generate coordinate of the future geometrical artifacts useful for plotting the data (the circles and the line that will separate the quadrants)
    #Generate circles coordinates
    circle_coords=np.array([[h0,w0,rays[i]] for i in range(n_rays)])
    #Generate lines
    rayout=rays[n_rays-1]
    rayin=rays[0]
    #Switch the comment next line if you want the rays converge through the center (the faidherbia)
    #line_coords=np.array([[h0,w0, h0+math.sin(tetas[i])*ray,w0+math.cos(tetas[i])*ray ] for i in range(n_tetas)])
    line_coords=np.array([[h0+math.sin(tetas[i])*rayin,w0+math.cos(tetas[i])*rayin, h0+math.sin(tetas[i])*rayout,w0+math.cos(tetas[i])*rayout ] for i in range(n_tetas)])

    #Remove crown
    crown_mask=radiusmap[:,:]<rays[1]
    stud=mask_studied_area[:,:]<1
    crown=crown_mask*stud
    polar_heatmap[crown]=valmin




    return result,polar_heatmap,circle_coords,line_coords

#This function take as input the outputs of the previous ones in order to draw a fancy polar heatmap, comparing original data and average data
def plot_polar_faidherbia(data,polar_heatmap,circle_coords,line_coords,h0,w0,path_to_figure_saving=None,valmin=0,valmax=0,colormap=None):

    #Define a default colormap
    if(colormap is None):
        colormap='viridis'

    n_circ=np.shape(circle_coords)[0]
    n_lines=np.shape(line_coords)[0]
    theta = np.linspace(0, 2*np.pi, 100)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    bigray=circle_coords[n_circ-1,2]

    #Draw black surrounding circles
    for cir in range(n_circ):
        ray=circle_coords[cir,2]
        x0=np.ones_like(theta)*circle_coords[cir,1]
        y0=np.ones_like(theta)*circle_coords[cir,0]
        ax1.plot(x0+ray*np.cos(theta), y0+ray*np.sin(theta), color='black')
        ax2.plot(x0+ray*np.cos(theta), y0+ray*np.sin(theta), color='black')

    #Draw radial lines
    for lin in range(n_lines):        
        ax1.plot([line_coords[lin,1],line_coords[lin,3]], [line_coords[lin,0],line_coords[lin,2]], color='black')
        ax2.plot([line_coords[lin,1],line_coords[lin,3]], [line_coords[lin,0],line_coords[lin,2]], color='black')

    #Set text
    delta=w0*0.1
    #    ax1.text(w0-delta,h0+bigray+delta+delta/4,'South',size=12,c='white')
    ax2.text(w0-delta,h0+bigray+delta+delta/4,'South',size=12,c='black')

    #    ax1.text(w0-delta,h0-bigray-delta+delta/4,'North',size=12,c='white')
    ax2.text(w0-delta,h0-bigray-delta+delta/4,'North',size=12,c='black')

    #    ax1.text(w0-bigray-2*delta,h0+delta/4,'West',size=12,c='white')
    ax2.text(w0-bigray-2*delta,h0+delta/4,'West',size=12,c='black')

    #    ax1.text(w0+bigray+delta/2,h0+delta/4,'East',size=12,c='white')
    ax2.text(w0+bigray+delta/2,h0+delta/4,'East',size=12,c='black')

#    ax1.text(w0-2*delta,h0+delta/4,' Faidherbia crown\n(no crops measured)',size=9,c='black')
    ax2.text(w0-3*delta,h0+delta/4,'               Crown',size=9,c='white')

    #Detect min / max
    delta=np.max(polar_heatmap)-np.min(polar_heatmap)
    print("delta="+str(delta))
    print("min="+str(np.min(polar_heatmap)))
    print("max="+str(np.max(polar_heatmap)))
    if(valmin==valmax):
        valmin=0.1#np.min(polar_heatmap)
        valmax=0.3#np.max(polar_heatmap)
    print("valmin="+str(valmin))
    print("valmax="+str(valmax))


    ax1.set(title='Original computed data')
    im1 = ax1.imshow(data, vmin=valmin,vmax=valmax,cmap=colormap)
    fig.colorbar(im1, ax=ax1, shrink=0.5)

    #Draw the polar heatmap
    ax2.set(title='Data averaged over quadrants')
    im2 = ax2.imshow(polar_heatmap, vmin=valmin,vmax=valmax,cmap=colormap)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    plt.show()
    #Save the figure as a tif file
    if(path_to_figure_saving is not None):
        fig.savefig(path_to_figure_saving,dpi=300)



def polar_viewer_on_actual_data(data,mask_studied_area,path_to_figure_saving=None,valmin=0.0,valmax=0.0,colormap=None):
    t0=time.perf_counter()

    original_height,original_width=np.shape(data)
    original_h0=original_height/2
    original_w0=original_width/2
    original_pix_size=1.30
    scale_factor=1

    h=original_height/scale_factor
    w=original_width/scale_factor
    pix_size=original_pix_size*scale_factor
    h0=original_h0/scale_factor
    w0=original_w0/scale_factor
    if(scale_factor != 1):
        data=resample_to_scale_factor(data,h,w)

    faidherbia_radius=400/pix_size # Based on data, 4m seems to be the mean radius of the Faidherbia over the pop
    delta=0
    radius_start=faidherbia_radius+delta
    rays=np.array([0.5*faidherbia_radius,1.5*faidherbia_radius,2.5*faidherbia_radius,3.5*faidherbia_radius,4.5*faidherbia_radius])
    print("")
    print("Rays used for binning are : "+str(rays))
    print("")

    print(" >> Starting. Time="+str(time.perf_counter()-t0))
    tetas=np.array([np.pi*0.125*i-np.pi for i in range(17)])
    print("")
    print("Angles used for binning are : "+str(tetas))

    print(" >> Time="+str(time.perf_counter()-t0))
    print("")

    results,polar_heatmap,circle_coords,line_coords=compute_polar_heatmap(data,h0,w0,rays,tetas,mask_studied_area=mask_studied_area,valmin=valmin)
#    polar_heatmap[mask_studied_area[:,:]<1]=valmin
    print(" >> Polar heatmap generated. Time="+str(time.perf_counter()-t0))
    print("")
    print("Average values computed for quadrants are : "+str(results))
    print("")
    plot_polar_faidherbia(data,polar_heatmap,circle_coords,line_coords,h0,w0,path_to_figure_saving,valmin,valmax,colormap)
    print(" >> Plot generated. Time="+str(time.perf_counter()-t0))


    


#Generate test data. It is a simple functional landscape, with a global minima somewhere near the center
def get_data(h,w):
    dat= np.array([[( np.abs(j-w/1.8)*np.abs(i-h/2.3) ) for j in range(w) ] for i in range(h)] )  # mat order = [T X Y F]
    max=np.max(np.max(dat))/2
    return dat/max

#This function could be used to run the test with very large images. 
def resample_to_scale_factor(tab,h,w):
    return resize(tab, (h,w), anti_aliasing=False)

def test_polar_viewer_on_synthetic_data():
    stress_factor=1
    #1 : 0.81
    #2 : 3.39
    #5 : 20
    #10: 81 'ou 50 en perf'
    t0=time.perf_counter()
    print(" >> Starting. Time="+str(time.perf_counter()-t0))
    h=1024*stress_factor
    w=1024*stress_factor

    tetas=np.array([np.pi*0.125*i-np.pi for i in range(17)])
    print("")
    print("Angles used for binning are : "+str(tetas))
    rays=np.array([h/5,h/4,h/3.3,h/2.75])
    print("")
    print("Rays used for binning are : "+str(rays))
    print("")

    if(False):
        data=get_data(h,w)
    else:
        data=np.random.rand(5,5)
        data=resample_to_scale_factor(data,h,w)
    print(" >> Data generated. Time="+str(time.perf_counter()-t0))
    print("")
    w0=w/2
    h0=h/2
    results,polar_heatmap,circle_coords,line_coords=compute_polar_heatmap(data,h0,w0,rays,tetas)
    print(" >> Polar heatmap generated. Time="+str(time.perf_counter()-t0))
    print("")
    print("Average values computed for quadrants are : "+str(results))
    print("")
    plot_polar_faidherbia(data,polar_heatmap,circle_coords,line_coords,0,1,h0,w0)
    print(" >> Plot generated. Time="+str(time.perf_counter()-t0))

#test_polar_viewer_on_actual_data()
