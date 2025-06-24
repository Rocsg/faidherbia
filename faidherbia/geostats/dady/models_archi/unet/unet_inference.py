import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import tifffile
import torch.nn.functional as F

# Ajouter la racine du projet au path
dady_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dady_root))

from utils.config_utils.path_utils import SQUARE_PATCHS_DIR
from models_archi.unet.unet_model import ChannelReconstructionUNet, masked_rmse_loss
from models_archi.unet.unet_data_loader import TifDataset

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChannelReconstructionUNet().to(device)
model_path = dady_root / "models_saved" / "reconstruction_model30.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

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
        loss = masked_rmse_loss(output, image_tensor, mask_tensor, invalid_tensor)
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


    return output, loss, image_tensor

img = Path("testdata/224x224_patchs/andrano10.tif")
mask = np.array([0, 1, 1, 0, 0], dtype=np.float32)  # Masque binaire
invalid = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # "invalid" channels = non présents dans l'image d'origine donc pas de calcul de perte
output, loss, image_tensor = infer_and_display(img, mask, invalid, display=False)

image_tensor = image_tensor.squeeze(0)
output = output.squeeze(0)

# Calcul du RMSE par canal reconstruit (selon le masque)
tmp = 0
for i in range(len(mask)):
    if mask[i] == 1:  # Canal reconstruit
        mse = F.mse_loss(image_tensor[i], output[i], reduction='mean')
        rmse = torch.sqrt(mse)
        print(f"Canal {i} - RMSE: {rmse.item():.4f}")
        tmp += rmse
print(tmp/np.sum(mask))

