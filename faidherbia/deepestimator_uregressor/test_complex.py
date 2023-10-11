import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torchvision import transforms
from PIL import Image
from faidherbia.deepestimator_uregressor.architecture_u_regressor import ResUNet
from torch.utils.data import Dataset, DataLoader
from faidherbia.datahandling.Olddataloader_simple_ndvi import CustomDataset
from faidherbia.datahandling.utils import get_working_directory


# Définir la fonction de test
def test_model(model):
    # Charger les données de test
    test_data_dir = get_working_directory() + 'Test_dataset'
    test_dataset = CustomDataset(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predicted_values = []  # Pour stocker les valeurs prédites
    true_values = []       # Pour stocker les valeurs réelles

    model.eval()  # Passer en mode d'évaluation (désactive le dropout, etc.)

    with torch.no_grad():
        for input_image, fcover_image, output_yield in test_loader:
            # Obtenir la valeur G prédite par le modèle
            fcover_output = model(input_image)
            plt.imshow(fcover_output[0, 0, :, :].detach().numpy())
            plt.show()
            plt.imshow(fcover_image[0, 0, :, :].detach().numpy())
            plt.show()

            predicted_value =0#g_output.item()
            true_value = output_yield.item()

            predicted_values.append(predicted_value)
            true_values.append(true_value)

    # Calculer le coefficient de détermination R²
    r_squared = r2_score(true_values, predicted_values)

    # Créer un scatterplot des valeurs prédites par rapport aux valeurs réelles
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, s=20, alpha=0.5)
    plt.xlabel('Ground Truth G')
    plt.ylabel('Predicted G')
    plt.title(f'Scatterplot (R² = {r_squared:.2f})')
    plt.grid(True)
    plt.show()

    return r_squared

# Utilisation de la fonction de test
model = ResUNet(in_channels=6, out_channels=1)
model.load_state_dict(torch.load('/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/Test_deep/Models/resunet_model.pth'))
r_squared = test_model(model)

print(f'R² sur les données de test : {r_squared:.2f}')
