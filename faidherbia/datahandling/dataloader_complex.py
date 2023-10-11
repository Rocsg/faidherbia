from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
from skimage import io

# Définir le jeu de données personnalisé (y compris les variables supplémentaires)
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir_input = data_dir+'/MultiSpectralImgs'
        self.data_dir_fcover = data_dir+'/FcoverDensityMap'
        self.data_dir_target_values = data_dir+'/TargetValues'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.image_list = os.listdir(self.data_dir_input)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #Open the input image
        img_name = os.path.join(self.data_dir_input, self.image_list[idx])
        image_input = io.imread(img_name)
        image_input = self.transform(image_input)
        image_input = image_input.permute(1, 2, 0)

        #Open the fcover image
        img_name = os.path.join(self.data_dir_fcover, self.image_list[idx])
        image_fcover = io.imread(img_name)
        image_fcover = self.transform(image_fcover)

        #Open the target value. It is a value located in column 2 of a csv file, in the line where the column 1 is the name of the image
        #Find the line where the name of the image is located, and collect the target value
        csv_name = os.path.join(self.data_dir_target_values, 'targetYield.csv')
        with open(csv_name, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.split(',')[0] == self.image_list[idx]:
                target_value = float(line.split(',')[1])
        target_value = torch.Tensor([target_value])            

        return image_input, image_fcover,target_value
