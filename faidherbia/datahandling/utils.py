import os
from skimage.transform import resize
import numpy as np
def squareOfA(a):
    return a**2

def get_working_directory():
    #If the file /home/rfernandez exists, return something
    if os.path.isdir('/home/rfernandez'):
        return '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/'
    else:
        return 'please mansour, set your working directory, with in it Full_dataset, TrainValid_dataset and Test_dataset'

def augment_image(image):
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

    #return a list of 8 images
    return [image1,image2,image3,image4,image5,image6,image7,image8]
