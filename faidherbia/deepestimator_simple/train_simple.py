import random
import elasticdeform.torch as elastic_transform
import torch
import torch.cuda as cuda
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import os
from PIL import Image
from faidherbia.deepestimator_simple.archi_regressor import ContractingResUNet
from faidherbia.datahandling.Olddataloader_simple_ndvi import CustomDataset
from faidherbia.datahandling.utils import get_working_directory
import tensorboardX
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


# Créez un SummaryWriter pour enregistrer les logs
log_dir = "/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Logs/Model_02"
writer = SummaryWriter(log_dir)


# Initialiser le modèle ResUNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ContractingResUNet(in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load('/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/simple_archi_model_02_02.pth'))
#Comment for building a new one

# Définir la fonction de perte pour F et G
criterion_yield = nn.MSELoss()
criterion_biomass = nn.MSELoss()

# Définir l'optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Charger les données
data_dir =     data_dir = get_working_directory() + 'TrainValid_dataset'

N = 10  # factor augmentation

custom_dataset = CustomDataset(data_dir,N=N)
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
batch_size = 12
train_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(custom_dataset, batch_size=1, sampler=val_sampler)

custom_dataset = CustomDataset(data_dir, N=N)
data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)
summary(model, (1,256, 256))

# Boucle d'entraînement avec validation
num_epochs = 50
for epoch in range(num_epochs):
    # Entraînement
    model.train()
    for input_image, target in train_loader:
        input_image = input_image.to(device)
        #print(np.shape(input_image))
        target = target.to(device)
        optimizer.zero_grad()
        estimated_values = model(input_image)
        #print("shape1")
        #print(np.shape(estimated_values))
        batch_len=np.shape(estimated_values)[0]
        estimated_values=estimated_values.view(batch_len, 2)
        estimated_yield=estimated_values[:,0]
        estimated_biomass=estimated_values[:,1]
        target_yield=target[:,0]
        target_biomass=target[:,1]

        loss_yield = criterion_yield(estimated_yield,target_yield)
        loss_biomass = criterion_biomass(estimated_biomass,target_biomass)
        loss_total = loss_yield+loss_biomass

        loss_total.backward()
#        loss_biomass.backward()
#        total_loss.backward()
        optimizer.step()
        # Enregistrez les pertes et autres statistiques dans TensorBoard
        writer.add_scalar('Loss biomass', loss_yield.item(), global_step=epoch)
        writer.add_scalar('Loss yield', loss_biomass.item(), global_step=epoch)
        writer.add_scalar('Loss total', loss_total.item(), global_step=epoch)


    # Validation
    model.eval()
    val_loss_yield = 0.0
    val_loss_biomass = 0.0
    val_loss_tot = 0.0
    with torch.no_grad():
        for input_image, target in train_loader:
            input_image = input_image.to(device)
            target = target.to(device)
            estimated_values = model(input_image)
            batch_len=np.shape(estimated_values)[0]
            estimated_values=estimated_values.view(batch_len, 2)
            estimated_yield=estimated_values[:,0]
            estimated_biomass=estimated_values[:,1]
            target_yield=target[:,0]
            target_biomass=target[:,1]

            loss_yield = criterion_yield(estimated_yield,target_yield)
            loss_biomass = criterion_biomass(estimated_biomass,target_biomass)
            loss_total = loss_yield+loss_biomass
            val_loss_biomass += loss_biomass.item()
            val_loss_yield += loss_yield.item()
            val_loss_tot += loss_total.item()

    # Calculer les moyennes des pertes de validation
    avg_val_loss_yield = val_loss_yield / len(val_loader)
    avg_val_loss_biomass = val_loss_biomass / len(val_loader)
    avg_val_loss_tot = val_loss_tot / len(val_loader)

    writer.add_scalar('Avg val loss yield', avg_val_loss_yield, global_step=epoch)
    writer.add_scalar('Avg val loss biomass', avg_val_loss_biomass, global_step=epoch)
    writer.add_scalar('Avg val loss total', avg_val_loss_tot, global_step=epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss_yield: {loss_yield.item()} Val_Loss_yield: {avg_val_loss_yield}, Loss_biomass: {loss_biomass.item()} Val_Loss_biomass: {avg_val_loss_biomass}, Loss_total: {loss_total.item()} Val_Loss_total: {avg_val_loss_tot} ')

writer.close()

# Enregistrez le modèle entraîné si nécessaire
torch.save(model.state_dict(), '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/simple_archi_model_02_03.pth')


#tensorboard --logdir=/chemin/vers/le/dossier/de/logs


