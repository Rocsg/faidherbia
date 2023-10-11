from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
from skimage import io, transform
import numpy as np
import random
import matplotlib.pyplot as plt
# Définir le nombre d'augmentations souhaité
num_augmentations = 10

#data_transforms = transforms.Compose([
#    transforms.RandomRotation(degrees=5),  # Rotation aléatoire de -30 à 30 degrés
#    transforms.RandomHorizontalFlip(p=0.5),  # Retournement horizontal avec une probabilité de 0.5
#    transforms.ToTensor(),
#])
# Liste des transformations que vous souhaitez appliquer
transformations = [
    transform.rotate,
    # Ajoutez d'autres transformations selon vos besoins
]


# Définir le jeu de données personnalisé (y compris les variables supplémentaires)
class CustomDataset(Dataset):
    def __init__(self, data_dir,N=1):
        self.data_dir_input = data_dir+'/MultiSpectralImgs'
        self.data_dir_target_values = data_dir+'/TargetValues'
        self.image_list = os.listdir(self.data_dir_input)
        self.N = N  # Nombre de répétitions pour chaque image source

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
        img_ndvi=(img_ir-img_r)/(img_ir+img_r)

#        if repeat_idx > 0:
#            random_transform = random.choice(transformations)
#            img_ndvi = random_transform(img_ndvi,0)
        #If the square is smaller than 256 pixels in some dimension, extend it by copying data on the other side
        if np.shape(img_ndvi)[0]<256:
            img_ndvi=np.concatenate((img_ndvi,img_ndvi),axis=0)
            img_ndvi=img_ndvi[0:256,:]
        if np.shape(img_ndvi)[1]<256:
            img_ndvi=np.concatenate((img_ndvi,img_ndvi),axis=1)
            img_ndvi=img_ndvi[:,0:256]


        #Select a square of 256 x 256 randomly in img_ndvi
        #Select the upper left corner of the square
        x = random.randint(0, np.shape(img_ndvi)[0]-256)
        y = random.randint(0, np.shape(img_ndvi)[1]-256)
        img_ndvi = img_ndvi[x:x+256,y:y+256]   
        
        init_surf=np.shape(img_ndvi)[0]*np.shape(img_ndvi)[1]

        #500 130
        max_biomasse=500
        max_yield=130
        patch_surf=65536

        #Open the target value. It is a value located in column 2 of a csv file, in the line where the column 1 is the name of the image
        #Find the line where the name of the image is located, and collect the target value
        csv_name = os.path.join(self.data_dir_target_values, 'Data_faidherbia.csv')
        with open(csv_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.split(',')[0]  == self.image_list[source_idx].split('.')[0]:
                target_yield = 1.0/(max_yield)*float(line.split(',')[8])*(patch_surf/init_surf)
                target_biomass = 1.0/(max_biomasse)*float(line.split(',')[9])*(patch_surf/init_surf)
        target_values = torch.Tensor([target_yield,target_biomass])            
        #Add an additionnal dimension to the image
        img_ndvi = np.expand_dims(img_ndvi, axis=0)

        return img_ndvi,target_values


#Test the dataset

#create a data_loader with N=3
data_dir = '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/TrainValid_dataset'
custom_dataset = CustomDataset(data_dir,N=10)
data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
#print("Data loader test")
#Use the dataloader to load image per image
if(False):
    for image,target in data_loader:
        #Print the target variables and show the patch
        print(target)
        plt.imshow(image[0,0,:,:].detach().numpy())
        plt.show()
        #wait for 2 seconds
        plt.pause(2)

        #Then close the plot
        plt.close()


