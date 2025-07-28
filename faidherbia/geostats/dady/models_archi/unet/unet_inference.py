import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import torch.optim as optim
import tifffile
import torch.nn.functional as F

# Ajouter la racine du projet au path
dady_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dady_root))

from utils.config_utils.path_utils import SQUARE_PATCHS_DIR
from tests.corrected_lossallchan import ChannelReconstructionUNet, masked_rmse_loss
from models_archi.unet.unet_data_loader import TifDataset

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChannelReconstructionUNet().to(device)

checkpoint_path = dady_root / "models_saved" / "x2ponderation_52.pth"


# Création du modèle + optimizer + scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Charger le checkpoint complet
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1
best_val_loss = checkpoint['best_val_loss']




def infer_and_display(img_path, mask_vector, invalid_vector, display=False):
    # Charger image (shape: [5, H, W])
    image = tifffile.imread(img_path)
    image = np.transpose(image, (2, 0, 1))  # → (5, H, W)
    
    # Appliquer le masque (mask_vector : [5])
    input_mask = 1.0 - mask_vector  # On garde les canaux non à reconstruire
    masked_image_np = image * input_mask[:, None, None]

    # Conversion en tensors + normalisation
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, 5, H, W)
    masked_tensor = torch.tensor(masked_image_np, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, 5, H, W)
    
    # CORRECTION: Convertir mask_vector en tensor PyTorch
    mask_tensor = torch.tensor(mask_vector, dtype=torch.float32).unsqueeze(0)  # (1, 5)
    invalid_tensor = torch.tensor(invalid_vector, dtype=torch.float32).unsqueeze(0)

    # Déplacer sur le device
    image_tensor = image_tensor.to(device)
    masked_tensor = masked_tensor.to(device)
    mask_tensor = mask_tensor.to(device)

    invalid_tensor = invalid_tensor.to(device)  

    with torch.no_grad():
        output = model(masked_tensor, mask_tensor)  
     
    if display:
        # Visualisation
        fig, axs = plt.subplots(5, 3, figsize=(12, 10))
        for i in range(5):
            axs[i, 0].imshow(image_tensor[0, i].cpu(), cmap='gray')
            axs[i, 0].set_title(f"Canal {i} - Original")

            axs[i, 1].imshow(masked_tensor[0, i].cpu(), cmap='gray')
            axs[i, 1].set_title(f"Canal {i} - Masqué")

            axs[i, 2].imshow(output[0, i].cpu(), cmap='gray') 
            axs[i, 2].set_title(f"Canal {i} - Reconstruit")

            for j in range(3):
                axs[i, j].axis("off")

        plt.tight_layout()
        plt.show()

    # Optionnel: calculer et afficher la perte
    with torch.no_grad():
        loss = masked_rmse_loss(output, image_tensor, invalid_tensor)
        print(f"RMSE Loss: {loss:.4f}")
    
        # === Statistiques par canal ===
    print("\n--- Statistiques par canal ---")
    def print_stats(tensor, name):
        data = tensor.squeeze(0).cpu().numpy()  # (5, H, W)
        print(f"\n{name}:")
        for i in range(5):
            chan = data[i]
            print(f"  Canal {i}: mean={chan.mean():.4f}, std={chan.std():.4f}, "
                  f"min={chan.min():.4f}, max={chan.max():.4f}")
    
    print_stats(image_tensor, "Image originale")
    print_stats(output, "Image reconstruite")


    return output, loss, image_tensor, image_tensor

img = Path(r"testdata\224x224_patchs\godetc1_1005.tif")
mask = np.array([1, 0, 0, 0, 1], dtype=np.float32)  # Masque binaire
invalid = np.array([1, 0, 0, 0, 0], dtype=np.float32)  # "invalid" channels = non présents dans l'image d'origine donc pas de calcul de perte
output, loss, image_tensor = infer_and_display(img, mask, invalid, display=True)

# === Nouveau : RMSE par canal ===
print("\n--- RMSE par canal ---")
for i in range(5):
    # Comparaison entre image originale et sortie reconstruite
    real = image_tensor[0, i].cpu().numpy()
    pred = output[0, i].cpu().numpy()
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    print(f"Canal {i} : RMSE = {rmse:.4f}")

def scatter_plot(image_tensor, output, channel_idx):
    
    real = image_tensor.squeeze(0)[channel_idx].cpu().numpy()
    pred = output.squeeze(0)[channel_idx].cpu().numpy()

    real_flat = real.flatten()
    pred_flat = pred.flatten()

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_flat, real_flat, alpha=0.3, s=1, color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='y = x')
    plt.xlabel("Valeur prédite (reconstruite)")
    plt.ylabel("Valeur réelle")
    plt.title(f"Scatter plot - Canal {channel_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation de la fonction scatter_plot
scatter_plot(image_tensor, output, channel_idx=4)


def plot_pixel_distribution(image_tensor, output, channel_idx, bins=255):
    """
    Affiche un histogramme comparatif des valeurs de pixels (0-1) entre
    l'image originale et l'image prédite pour un canal donné.

    :param image_tensor: Tensor de l'image originale (1, 5, H, W)
    :param output: Tensor de l'image prédite (1, 5, H, W)
    :param channel_idx: Indice du canal (0 à 4)
    :param bins: Nombre de bins pour l'histogramme
    """
    real = image_tensor.squeeze(0)[channel_idx].cpu().numpy().flatten()
    pred = output.squeeze(0)[channel_idx].cpu().numpy().flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(real, bins=bins, alpha=0.5, color='green', label='Original', range=(0, 1), density=False)
    plt.hist(pred, bins=bins, alpha=0.5, color='blue', label='Reconstruit', range=(0, 1), density=False)

    plt.xlabel("Valeur des pixels (0 à 1)")
    plt.ylabel("Nombre de pixels")
    plt.title(f"Histogramme des valeurs de pixel - Canal {channel_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_pixel_distribution(image_tensor, output, channel_idx=0)

import imageio

from tifffile import imwrite
import tifffile

def save_channel_as_tif(image_tensor, output, channel_idx, save_dir="outputs", base_name="sample"):
    """
    Sauvegarde un canal original et reconstruit en tant qu'images float32 TIFs, normalisées [0, 1]
    """
    from pathlib import Path
    import os
    os.makedirs(save_dir, exist_ok=True)

    real = image_tensor[0, channel_idx].cpu().numpy()
    pred = output[0, channel_idx].cpu().numpy()

    real_path = Path(save_dir) / f"{base_name}_channel{channel_idx}_original.tif"
    pred_path = Path(save_dir) / f"{base_name}_channel{channel_idx}_predicted.tif"

    # Sauvegarder en float32 TIF sans conversion en uint8
    tifffile.imwrite(real_path, real.astype(np.float32))
    tifffile.imwrite(pred_path, pred.astype(np.float32))

    print(f"Images sauvegardées (float32): {real_path} et {pred_path}")


save_channel_as_tif(image_tensor, output, channel_idx=0, save_dir="outputs", base_name="andrano10")
