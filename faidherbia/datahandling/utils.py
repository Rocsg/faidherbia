import os
from skimage.transform import resize
import numpy as np
import random

def nbPlacettes():
    return 12

def squareOfA(a):
    return a**2

def get_working_directory():
    #If the file /home/rfernandez exists, return something
    if os.path.isdir('/home/rfernandez'):
        return '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/'
    else:
        return '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/'

def augment_image(image,swapping):
    #resize image to get 256x256 image
    image = resize(image, (256, 256))

    #create 8 images by rotating the image by 90° and making mirror of it
    image1 = image
    image2 = np.rot90(image1)
    image3 = np.rot90(image2)
    image4 = np.rot90(image3)
    image5 = np.flipud(image1)
    image6 = np.flipud(image2)
    image7 = np.flipud(image3)
    image8 = np.flipud(image4)

    vect=[image1,image2,image3,image4,image5,image6,image7,image8]
    if(swapping):
        for i in range(8):
            if(random.random() < 0.6):
                vect[i]=interchange_patches(vect[i])

    #return a list of 8 images
    return vect



def interchange_patches(image):
    rows, cols = image.shape[0], image.shape[1]
    patch_size = 64  # Size of each 3x3 patch

    # Randomly select two different patches
    patch_indices = list(range(9))
    random.shuffle(patch_indices)
    patch1, patch2 = patch_indices[:2]

    # Calculate the starting coordinates of the selected patches
    row1, col1 = (patch1 // 3) * patch_size, (patch1 % 3) * patch_size
    row2, col2 = (patch2 // 3) * patch_size, (patch2 % 3) * patch_size

    # Swap the patches
    image[row1:row1 + patch_size, col1:col1 + patch_size], image[row2:row2 + patch_size, col2:col2 + patch_size] = image[row2:row2 + patch_size, col2:col2 + patch_size], image[row1:row1 + patch_size, col1:col1 + patch_size]

    return image