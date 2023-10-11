import time
from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,explained_variance_score

import torch
import torch.cuda as cuda
import torch.nn as nn
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import elasticdeform.torch as elastic_transform
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,  SequentialSampler
import torchmetrics
from torchmetrics.functional import r2_score as r2
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from faidherbia.datahandling.dataloader_simple_aug import CustomDataset
from faidherbia.datahandling.utils import get_working_directory
import tensorboardX
from tensorboardX import SummaryWriter
# Créez un SummaryWriter pour enregistrer les logs

exp="Model_dense_12_CyclicLR_R2target"
log_dir = "/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Logs/"+exp
writer = SummaryWriter(log_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

grayscale=False


# Définissez les valeurs minimale et maximale du taux d'apprentissage et la durée d'un cycle (en nombre d'epochs)
min_lr = 0.0003
max_lr = 0.0004
step_size = 20  # La moitié de la période d'un cycle (nombre d'epochs pour une montée puis une descente)
batch_size = 24
num_epochs=100



use_resnet=True
if(use_resnet):
    densenet = models.resnet50(pretrained=True)
    # Supprimer la couche de classification par défaut
    num_features = densenet.fc.in_features
    densenet.fc = nn.Identity()
    # Ajouter une couche de régression personnalisée
    regression_layer = nn.Sequential(
        nn.Linear(num_features, 128),  # Vous pouvez ajuster la taille de la couche cachée
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2)  # Pour une régression univariée. Utilisez 2 pour deux valeurs de régression, etc.
    )
    if(grayscale): 
        densenet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

use_dense=False
if(use_dense):
    # Load pre-trained DenseNet-201 model
    densenet = models.densenet201(pretrained=True)
    # Modify the input layer to accept 1-channel grayscale images
    #Unactivated for three channels images
    if(grayscale): 
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    # Modify the output layer for regression (double neuron with linear activation)
    #densenet.classifier = nn.Linear(1920, 2)
    densenet.classifier = nn.Sequential(
        nn.Linear(1920, 2)  # Apply sigmoid activation for each output
    )


use_mobilenet=False
if(use_mobilenet):
    from torchvision.models import mobilenet_v3_small

    # Load the pre-trained MobileNetV3 model (you can also use 'mobilenet_v3_small' if needed)
    densenet = mobilenet_v3_small(pretrained=True)

    # Freeze feature layers
    for param in densenet.parameters():
        param.requires_grad = False

    # Modify the classifier for regression
    densenet.classifier[3] = nn.Sequential(
        nn.Linear(1024, 2),  # Adjust the input size according to your specific MobileNetV3 variant
    )


# Define the loss function (e.g., Mean Squared Error)
criterion = nn.MSELoss()
# Define an optimizer (e.g., Adam)
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)


N = 8  # factor augmentation
data_dir =     data_dir = get_working_directory() + 'TrainValid_dataset'
custom_dataset = CustomDataset(data_dir,N=N,grayscale=grayscale)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=False)



# Définir le ratio de division (par exemple, 80% pour l'entraînement, 20% pour la validation)
validation_split = 0.7
nb_images=len(custom_dataset)/N
dataset_size = len(custom_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * nb_images*N))

# Diviser les indices en ensembles d'entraînement et de validation
train_indices, val_indices = indices[split:], indices[:split]
random.shuffle(train_indices)
random.shuffle(val_indices)


# Créez un planificateur de taux d'apprentissage cyclique
clr_scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=step_size, mode='exp_range',cycle_momentum=False)

# Créer des échantillonneurs pour les ensembles d'entraînement et de validation
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SequentialSampler(val_indices)


custom_dataset = CustomDataset(data_dir,N=N,grayscale=grayscale)
train_loader = DataLoader(custom_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(custom_dataset, batch_size=8, sampler=val_sampler)
densenet=densenet.to(device)
val_loss_save=100000





start_time = time.time()
tot_elapsed_time=0

########### TRAINING LOOP ################
for epoch in range(num_epochs):
    # Initialize variables to keep track of metrics for this epoch
#    densenet.zero_grad()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    densenet.train()
    true_y=[]
    true_b=[]
    predicted_y=[]
    predicted_b=[]
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if np.shape(targets)[0]==1:
            continue
        targets = targets.to(device)
        #print(targets)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = densenet(inputs)
        loss = 1-r2(outputs[:,0], targets[:,0])-r2(outputs[:,1], targets[:,1])
        loss.backward()
        optimizer.step()

        target_yield=targets[:,0]
        target_biomass=targets[:,1]
        true_y.extend(target_yield.detach().cpu().numpy())    
        true_b.extend(target_biomass.detach().cpu().numpy())

        estimated_yield=outputs[:,0]
        estimated_biomass=outputs[:,1]
        predicted_y.extend(estimated_yield.detach().cpu().numpy())
        predicted_b.extend(estimated_biomass.detach().cpu().numpy())

    clr_scheduler.step()

    r_squared_y = r2_score(np.array(true_y), np.array(predicted_y))
    r_squared_b = r2_score(np.array(true_b), np.array(predicted_b))
    train_loss=1-r_squared_y-r_squared_b
    r2_mean_train=(r_squared_y+r_squared_b)/2
    writer.add_scalar('Train loss', train_loss, global_step=epoch)
    writer.add_scalar('R2mean_Train', r2_mean_train, global_step=epoch)


    true_y=[]
    true_b=[]
    predicted_y=[]
    predicted_b=[]
    epoch_loss = 0.0
    num_batches = len(val_loader)
    densenet.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            #if targets first dimension is 1, skip to the next iteration
            if np.shape(targets)[0]==1:
                continue
            targets = targets.to(device)
            inputs = inputs.to(device)
            outputs = densenet(inputs)
 
            #a=the mean of values of the first column of the tensor targets
            
            target_yield=np.mean(targets[:,0].detach().cpu().numpy())
            target_biomass=np.mean(targets[:,1].detach().cpu().numpy())
            estimated_yield=np.mean(outputs[:,0].detach().cpu().numpy())
            estimated_biomass=np.mean(outputs[:,1].detach().cpu().numpy())

            #add target_yield to the list true_y
            


            true_y.extend([target_yield])    
            true_b.extend([target_biomass])
            predicted_y.extend([estimated_yield])
            predicted_b.extend([estimated_biomass])

    r_squared_y = r2_score(np.array(true_y), np.array(predicted_y))
    r_squared_b = r2_score(np.array(true_b), np.array(predicted_b))
    val_loss=1-r_squared_y-r_squared_b
    r2_mean_val=(r_squared_y+r_squared_b)/2
    writer.add_scalar('Val loss', train_loss, global_step=epoch)
    writer.add_scalar('R2mean_Val', r2_mean_val, global_step=epoch)


    #If improving val loss, then save the current model in a snapshot
    if val_loss<val_loss_save:
        torch.save(densenet.state_dict(), '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/'+exp+'_SNAPSHOT.pth')
        print(">>>>>>>> Snapshot saved when reaching "+str(val_loss)+' lower than '+str(val_loss_save)+' <<<<<<<<<')
        val_loss_save=val_loss

    writer.add_scalar('Val loss', val_loss, global_step=epoch)
    writer.add_scalar('R2Y', r_squared_y, global_step=epoch)
    writer.add_scalar('R2B', r_squared_b, global_step=epoch)
    writer.add_scalar('Learning rate', clr_scheduler.get_last_lr()[0], global_step=epoch)


    #Si epoch est multiple de 10, calculer le temps écoulé pour 10 epochs et estimer le temps restant en utilisant le timer
    if (epoch+1)%10==0:
        tot_elapsed_time=time.time() - start_time
        print("")
        print("------> Elapsed time for "+str(epoch+1)+" epochs: "+str(tot_elapsed_time)+" - Remaining time: "+str((num_epochs-(epoch+1))*tot_elapsed_time*(1.0/(epoch+1)))+"<-------")
        print("")

    # Calculer le coefficient de détermination R²
    r_squared_y = r2_score(true_y, predicted_y)



# Vous pouvez ajouter ici le code pour surveiller les métriques, comme le montant du taux d'apprentissage
    print(f"Epoch [{epoch + 1}/{num_epochs}]  TrLoss: {train_loss:.4f} TrR2: {r2_mean_train:.4f} | Val_loss: {val_loss:.4f} Val_R2: {r2_mean_val:.4f} | R2Y : {r_squared_y} - R2B : {r_squared_b} Learning Rate: {clr_scheduler.get_last_lr()[0]:.6f}")
 
writer.close()
torch.save(densenet.state_dict(), '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/'+exp+'.pth')

print("Evaluate : ")
print("tensorboard --logdir="+log_dir)
