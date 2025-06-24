from pathlib import Path
import numpy as np
import sys
import tifffile
from itertools import combinations
import torch
# Ajouter la racine du projet au path
dady_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dady_root))

from utils.config_utils.path_utils import SQUARE_PATCHS_DIR
from models_archi.unet.unet_model import ChannelReconstructionUNet, masked_rmse_loss
from models_archi.unet.unet_data_loader import TifDataset
from models_archi.unet.unet_inference import infer_and_display


def generate_binary_combinations(n=5):
    vectors = []

    # Combinaisons avec exactement 1 bit à 1
    for idx in combinations(range(n), 1):
        vec = np.zeros(n, dtype=int)
        vec[list(idx)] = 1
        vectors.append(vec)

    # Combinaisons avec exactement 2 bits à 1
    for idx in combinations(range(n), 2):
        vec = np.zeros(n, dtype=int)
        vec[list(idx)] = 1
        vectors.append(vec)

    return np.array(vectors)

# Exemple d'utilisation
all_vectors = generate_binary_combinations()

directory = Path(SQUARE_PATCHS_DIR)
tif_files = list(directory.glob(f"{"andrano"}*.tif")) + list(directory.glob(f"{"sahel"}*.tiff"))
results = {}

for vec in all_vectors:
    loss_tmp = 0
    invalid = np.zeros(5, dtype=int)

    for tif_file in tif_files:
        _, loss, _ = infer_and_display(tif_file, vec, invalid)
        loss_tmp += loss
    results[tuple(vec)] = loss_tmp / len(tif_files)
 
print("RMSE per masked channel: ", results)

        
import matplotlib.pyplot as plt

# Préparer les données pour l'affichage
labels = [''.join(map(str, k)) for k in results.keys()]  # '01010', etc.
values = [v.item() if isinstance(v, torch.Tensor) else v for v in results.values()]

# Créer le graphique
plt.figure(figsize=(15, 6))
bars = plt.bar(range(len(labels)), values, color='skyblue')
plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel("Combinaisons de canaux masqués (1 = à reconstruire)")
plt.ylabel("RMSE moyen")
plt.title("RMSE par combinaison de canaux masqués")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Affichage des valeurs sur les barres (facultatif)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", 
             ha='center', va='bottom', fontsize=8)

plt.show()
