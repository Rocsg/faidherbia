import random
import elasticdeform.torch as elastic_transform
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
from faidherbia.deepestimator_uregressor.architecture_u_regressor import ResUNet
from faidherbia.datahandling.Olddataloader_simple_ndvi import CustomDataset
from faidherbia.datahandling.utils import get_working_directory
import matplotlib.pyplot as plt

# Initialiser le modèle ResUNet
model = ResUNet(in_channels=6, out_channels=1)

# Définir la fonction de perte pour F et G
criterion_F = nn.MSELoss()
criterion_G = nn.MSELoss()

# Définir l'optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Charger les données
data_dir =     data_dir = get_working_directory() + 'TrainValid_dataset'

custom_dataset = CustomDataset(data_dir)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Définir le ratio de division (par exemple, 80% pour l'entraînement, 20% pour la validation)
validation_split = 0.5
dataset_size = len(custom_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
random.shuffle(indices)

# Diviser les indices en ensembles d'entraînement et de validation
train_indices, val_indices = indices[split:], indices[:split]

# Créer des échantillonneurs pour les ensembles d'entraînement et de validation
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Créer des chargeurs de données pour l'entraînement et la validation
batch_size = 1
train_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=val_sampler)




# Boucle d'entraînement avec validation
num_epochs = 50
for epoch in range(num_epochs):
    # Entraînement
    model.train()
    for input_image, fcover_image, output_yield in train_loader:
        optimizer.zero_grad()
        #f_output, g_output = model(input_image)
        f_output = model(input_image)

        # Calculer les pertes F et G
        loss_F = criterion_F(f_output, fcover_image)
        #loss_G = criterion_G(g_output, output_yield)

        # Combinez les pertes avec des poids
        weight_F = 0.5  # Ajustez ces poids selon vos besoins
        #weight_G = 0.5
        total_loss = weight_F * loss_F# + weight_G * loss_G

        total_loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss_F = 0.0
    #val_loss_G = 0.0
    with torch.no_grad():
        for input_image, fcover_image, output_yield in val_loader:
            f_output = model(input_image)

            # Calculer les pertes F et G sur l'ensemble de validation
            loss_F = criterion_F(f_output, fcover_image)
            #loss_G = criterion_G(g_output, output_yield)

            val_loss_F += loss_F.item()
            #val_loss_G += loss_G.item()

    # Calculer les moyennes des pertes de validation
    avg_val_loss_F = val_loss_F / len(val_loader)
    #avg_val_loss_G = val_loss_G / len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss_F: {loss_F.item()} Val_Loss_F: {avg_val_loss_F}')

# Enregistrez le modèle entraîné si nécessaire
torch.save(model.state_dict(), '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/resunet_model.pth')




