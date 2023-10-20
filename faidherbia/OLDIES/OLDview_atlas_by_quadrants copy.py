import numpy as np
import math
import matplotlib.pyplot as plt
import time
from skimage.transform import resize

#Create a copy of a numpy array, then replace each value to map the distance from the pixel to the target point (h0,w0) 
def get_r_map(h0,w0,data):
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
def compute_polar_heatmap(data,h0,w0,rays,tetas):
    #Prepare variables for the algorithm
    h=np.shape(data)[0]
    w=np.shape(data)[1]
    tetamap=get_teta_map(h0,w0,data)
    rmap=get_r_map(h0,w0,data)
    n_rays=len(rays)
    n_tetas=len(tetas)
    result=np.zeros((len(rays),len(tetas)))
    polar_heatmap=np.zeros_like(data)

    #Set the value to zero inside the smaller circle, indicating the out-of-study area (where the tree is, for example)
    mask=rmap[:,:]<rays[0]
    val_out=0
    result[0,:len(tetas)]=val_out
    polar_heatmap=polar_heatmap+mask*val_out

    #Set the value outside of the bigger circle, by computing the average value of all points lying outside this circle. It is assumed to be mean value "elsewhere"
    mask=rmap[:,:]>rays[n_rays-1]
    val_out=np.nanmean(data[mask])
    result[n_rays-1,:]=val_out
    polar_heatmap=polar_heatmap+mask*val_out

    #Compute the average value for each quadrant in the area under study
    for nr in range(n_rays-1):
        r1=rays[nr]
        r2=rays[nr+1]
        #These are the points lying in this ray interval
        mask_r=(rmap[:,:]>=r1) & (rmap[:,:]<r2)

        for nt in range(n_tetas):
            t1=tetas[nt]
            t2=1000
            if (nt!=(n_tetas-1)):
                t2=tetas[nt+1]
            #These are the points lying in this teta interval
            mask_t=(tetamap[:,:]>=t1) & (tetamap[:,:]<t2)

            #Combine ray interval and teta interval, and compute the average over these points
            mask=mask_r*mask_t
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
    return result,polar_heatmap,circle_coords,line_coords

#This function take as input the outputs of the previous ones in order to draw a fancy polar heatmap, comparing original data and average data
def plot_polar_faidherbia(data,polar_heatmap,circle_coords,line_coords,valmin,valmax,h0,w0):
    n_circ=np.shape(circle_coords)[0]
    n_lines=np.shape(line_coords)[0]
    theta = np.linspace(0, 2*np.pi, 100)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    bigray=circle_coords[n_circ-1,2]

    #Draw circles
    for cir in range(n_circ):
        ray=circle_coords[cir,2]
        x0=np.ones_like(theta)*circle_coords[cir,1]
        y0=np.ones_like(theta)*circle_coords[cir,0]
        ax1.plot(x0+ray*np.cos(theta), y0+ray*np.sin(theta), color='black')
        ax2.plot(x0+ray*np.cos(theta), y0+ray*np.sin(theta), color='black')

    #Draw lines
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
    ax2.text(w0-3*delta,h0+delta/4,'      Faidherbia crown',size=9,c='white')

    #Draw original data
    ax1.set(title='Original computed data')
    im1 = ax1.imshow(data, vmin=valmin,vmax=valmax)
    fig.colorbar(im1, ax=ax1, shrink=0.5)

    #Draw the polar heatmap
    ax2.set(title='Data averaged over quadrants')
    im2 = ax2.imshow(polar_heatmap, vmin=valmin,vmax=valmax)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    plt.show()



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

def test_polar_viewer_on_actual_data():
    data=np.load('/home/rfernandez/Bureau/A_Test/Mansour/droite.npy')
    valmin=0.3
    valmax=0.5
    t0=time.perf_counter()

    original_pix_size=1.30
    original_height,original_width=np.shape(data)
    original_h0=original_height/2
    original_w0=original_width/2
#    original_w0=3000
#    original_h0=1920

    scale_factor=1
    h=original_height/scale_factor
    w=original_width/scale_factor
    h0=original_h0/scale_factor
    w0=original_w0/scale_factor
    pix_size=original_pix_size*scale_factor
    data=resample_to_scale_factor(data,h,w)

    faidherbia_radius=400/pix_size
    delta=0
    radius_start=faidherbia_radius+delta
    rays=np.array([1.5*faidherbia_radius,2.5*faidherbia_radius,3.5*faidherbia_radius,4.5*faidherbia_radius])
    print("")
    print("Rays used for binning are : "+str(rays))
    print("")

    print(" >> Starting. Time="+str(time.perf_counter()-t0))
    tetas=np.array([np.pi*0.125*i-np.pi for i in range(17)])
    print("")
    print("Angles used for binning are : "+str(tetas))

    print(" >> Data generated. Time="+str(time.perf_counter()-t0))
    print("")

    results,polar_heatmap,circle_coords,line_coords=compute_polar_heatmap(data,h0,w0,rays,tetas)
    print(" >> Polar heatmap generated. Time="+str(time.perf_counter()-t0))
    print("")
    print("Average values computed for quadrants are : "+str(results))
    print("")
    plot_polar_faidherbia(data,polar_heatmap,circle_coords,line_coords,valmin,valmax,h0,w0)
    print(" >> Plot generated. Time="+str(time.perf_counter()-t0))



#data=np.load('/home/rfernandez/Bureau/A_Test/Mansour/faidh_mean_42faidhv2.npy')
#delta=60
#data2=data[0,6080:6080+3808,3816+delta:3816+3808+delta]
#np.save('/home/rfernandez/Bureau/A_Test/Mansour/data.npy',data2)
test_polar_viewer_on_actual_data()

#data2=np.load('/home/rfernandez/Bureau/A_Test/Mansour/data.npy')
#plt.imshow(data2)
#plt.show() 
#print(np.shape(data2))
if(False):
    test_polar_viewer_on_synthetic_data()
if(False):
    data=np.random.rand(500,500)
    data2=resample_to_scale_factor(data,5000,5000)
    data3=resample_to_scale_factor(data,500,500)
    print(np.mean(data))
    print(np.mean(data2))
    print(np.mean(data3))
    valmin=-1
    valmax=1
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    im1 = ax1.imshow(data2, vmin=valmin,vmax=valmax)
    fig.colorbar(im1, ax=ax1, shrink=0.5)
    im2 = ax2.imshow(data3, vmin=valmin,vmax=valmax)
    fig.colorbar(im2, ax=ax2, shrink=0.5)
    plt.show()