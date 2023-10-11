import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torchvision import transforms
from PIL import Image

import torch.nn as nn
from faidherbia.datahandling.dataloader_simple_aug import CustomDataset
from torch.utils.data import Dataset, DataLoader
from faidherbia.datahandling.utils import get_working_directory
import numpy as np
import torchvision.models as models

# Définir la fonction de test
def test_model():
    exp="Model_dense_10_CyclicLR_R2target"

    # Load pre-trained DenseNet-201 model
    densenet = models.densenet201(pretrained=True)
    # Modify the input layer to accept 1-channel grayscale images
    #densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Modify the output layer for regression (double neuron with linear activation)
    densenet.classifier = nn.Linear(1920, 2)
    # Define the loss function (e.g., Mean Squared Error)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    densenet.load_state_dict(torch.load('/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/'+exp+'_SNAPSHOT.pth'))
    densenet=densenet.to(device)

    # Charger les données de test
    test_data_dir = get_working_directory() + 'Test_dataset'
    test_dataset = CustomDataset(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    predicted_y = []  # Pour stocker les valeurs prédites
    true_y = []       # Pour stocker les valeurs réelles
    predicted_b = []  # Pour stocker les valeurs prédites
    true_b = []       # Pour stocker les valeurs réelles

    densenet.eval()  # Passer en mode d'évaluation (désactive le dropout, etc.)

    with torch.no_grad():
        for input_image, target in test_loader:
            input_image = input_image.to(device)
            target = target.to(device)
           
    


            estimated_values = densenet(input_image)
    
    
            target_yield=np.median(target[:,0].detach().cpu().numpy())
            target_biomass=np.median(target[:,1].detach().cpu().numpy())
            estimated_yield=np.median(estimated_values[:,0].detach().cpu().numpy())
            estimated_biomass=np.median(estimated_values[:,1].detach().cpu().numpy())

            #add target_yield to the list true_y
            


            true_y.extend([target_yield])    
            true_b.extend([target_biomass])
            predicted_y.extend([estimated_yield])
            predicted_b.extend([estimated_biomass])

    

    # Calculer le coefficient de détermination R²
    r_squared_y = r2_score(true_y, predicted_y)
    print(true_y)
    print(predicted_y)
    print(abs(np.array(true_y)-np.array(predicted_y))/np.array(true_y))
    # Créer un scatterplot des valeurs prédites par rapport aux valeurs réelles
    plt.figure(figsize=(8, 8))
    plt.scatter(true_y, predicted_y, s=20, alpha=0.5)
    plt.xlabel('Ground Truth Y')
    plt.ylabel('Predicted Y')
    plt.title(f'Scatterplot (R² = {r_squared_y:.2f})')
    plt.grid(True)
    plt.show()

    # Calculer le coefficient de détermination R²
    r_squared_b = r2_score(true_b, predicted_b)

    # Créer un scatterplot des valeurs prédites par rapport aux valeurs réelles
    plt.figure(figsize=(8, 8))
    plt.scatter(true_b, predicted_b, s=20, alpha=0.5)
    plt.xlabel('Ground Truth Y')
    plt.ylabel('Predicted Y')
    plt.title(f'Scatterplot (R² = {r_squared_b:.2f})')
    plt.grid(True)
    plt.show()

    return r_squared_y,r_squared_b

# Utilisation de la fonction de test
test_model()
