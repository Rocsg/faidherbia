from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
from skimage import io, transform
import numpy as np
import random
import matplotlib.pyplot as plt
from faidherbia.datahandling.utils import augment_image,nbPlacettes,squareOfA

# Définir le nombre d'augmentations souhaité
num_augmentations = 8
#Define the test data_dir
data_dir = '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/TrainValid_dataset'
testing=False
swapping=False
# Définir le jeu de données personnalisé (y compris les variables supplémentaires)
class CustomDataset(Dataset):
    def __init__(self, data_dir,N=num_augmentations,grayscale=True,swapping=False):
        self.swapping=swapping
        self.data_dir_input = data_dir+'/MultiSpectralImgs'
        self.data_dir_target_values = data_dir+'/TargetValues'
        self.image_list = os.listdir(self.data_dir_input)
        self.N = N  # Nombre de répétitions pour chaque image source
        self.grayscale=grayscale

    def __len__(self):
        return len(self.image_list) * self.N


    def __getitem__(self, idx):

        # Calculer l'indice de l'image source et le numéro de répétition
        source_idx = idx // self.N
        repeat_idx = idx % self.N

        #Open the input image
        img_name = os.path.join(self.data_dir_input, self.image_list[source_idx])
        image_input = io.imread(img_name)
        img_ir=image_input[:,:,4]
        img_r=image_input[:,:,2]
        img_n=(img_ir-img_r)/(img_ir+img_r)
        img_b=image_input[:,:,1]
        
        tab_imgs=augment_image(img_n,self.swapping)
        img_n=tab_imgs[repeat_idx].copy()
        tab_imgs=augment_image(img_r,self.swapping)
        img_r=tab_imgs[repeat_idx].copy()
        tab_imgs=augment_image(img_b,self.swapping)
        img_b=tab_imgs[repeat_idx].copy()

        max_biomasse=500
        max_yield=130

        #Open the target value. It is a value located in column 2 of a csv file, in the line where the column 1 is the name of the image
        #Find the line where the name of the image is located, and collect the target value
        csv_name = os.path.join(self.data_dir_target_values, 'Data_faidherbia.csv')
        with open(csv_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.split(',')[0]  == self.image_list[source_idx].split('.')[0]:
                target_yield = 1.0/(max_yield)*float(line.split(',')[8])
                target_biomass = 1.0/(max_biomasse)*float(line.split(',')[9])
                target_micro1=float(line.split(',')[4])
                target_micro2=float(line.split(',')[5])
                target_micro3=float(line.split(',')[6])
                target_micro4=float(line.split(',')[7])
                target_micro5=float((line.split(',')[0]).split('A')[0].replace('P',''))
        target_values = torch.Tensor([target_yield,target_biomass])            
        placettes=np.zeros((nbPlacettes()))
        placettes[int(target_micro5)-1]=1

        micro_values  = [target_micro1,target_micro2,target_micro3,target_micro4]
        # Fuse placettes and micro-values into a single vector
        micro_values = np.concatenate((placettes, micro_values))
        micro_values = torch.Tensor(micro_values)
        #Add an additionnal dimension to the image
        if(self.grayscale):
            img_ndvi = np.expand_dims(img_n, axis=0)
        else:
            #stack img_r, img_n and img_b
            img_ndvi=np.stack((img_r,img_n,img_b),axis=0)
        #Convert the image to a torch tensor
        img_ndvi = torch.from_numpy(img_ndvi)
        #make a pause of 1000 seconds

 
        return img_ndvi,target_values,micro_values


#Test the dataset

#create a data_loader with N=3
grayscale=True
custom_dataset = CustomDataset(data_dir,N=8,grayscale=grayscale,swapping=True)
data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
#print("Data loader test")
#Use the dataloader to load image per image
if(testing):
    for image,target,microvar in data_loader:
        #Print the target variables and show the patch
        print(target)
        print(microvar)
        print(image)
        print(image.shape)
        plt.imshow(image[0,0,:,:].detach().numpy())
        plt.show()
        plt.close()
        if(not grayscale):
            plt.imshow(image[0,1,:,:].detach().numpy())
            plt.show()
            plt.close()
            plt.imshow(image[0,2,:,:].detach().numpy())
            plt.show()
            plt.close()
            plt.imshow(image[0,0,:,:].detach().numpy()-image[0,2,:,:].detach().numpy())
            plt.show()
            plt.close()

        
