from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
from torch.utils.data import DataLoader
import numpy as np
from faidherbia.datahandling.utils import get_working_directory
from faidherbia.datahandling.Olddataloader_simple_ndvi import CustomDataset

data_dir =     data_dir = get_working_directory() + 'TrainValid_dataset'
custom_dataset = CustomDataset(data_dir,N=1)
dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=False)

# Initialisez les accumulateurs pour la moyenne et l'écart type
min_accumulator = 0.0
max_accumulator = 0.0

# Compteur pour suivre le nombre d'exemples
count = 0

# Parcourez le DataLoader pour collecter les statistiques
for image,value in dataloader:
    print(count)
    
    # Calculez la moyenne et l'écart type du mini-batch
    min_accumulator += image.min()
    max_accumulator += image.max()
    
    count += 1

# Calculez la moyenne et l'écart type global sur l'ensemble du dataset
min_global = min_accumulator / count
max_global = max_accumulator / count

print(min_global)
print(max_global)

#tensor(-0.1107)
#tensor(0.4079)
