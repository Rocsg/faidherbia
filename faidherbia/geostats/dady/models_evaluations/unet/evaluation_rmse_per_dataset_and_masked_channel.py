from pathlib import Path
import numpy as np
import sys
import tifffile
from itertools import combinations

# Ajouter la racine du projet au path
dady_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(dady_root))

from utils.config_utils.path_utils import SQUARE_PATCHS_DIR
from tests.corrected_lossallchan import ChannelReconstructionUNet, masked_rmse_loss
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

def generate_binary_combinations_uc1(n=5):
    vectors = []
    vec = np.zeros(n, dtype=int)
    vec[0] = 1
    vectors.append(vec)

    # Cas où canal 0 + 1 autre canal sont masqués
    for idx in combinations(range(1, n), 1):
        vec = np.zeros(n, dtype=int)
        vec[0] = 1
        vec[list(idx)] = 1
        vectors.append(vec)

    # # Cas où canal 0 + 2 autres canaux sont masqués
    # for idx in combinations(range(1, n), 2):
    #     vec = np.zeros(n, dtype=int)
    #     vec[0] = 1
    #     vec[list(idx)] = 1
    #     vectors.append(vec)

    return np.array(vectors)


# Exemple d'utilisation
all_vectors = generate_binary_combinations()
print(all_vectors)
directory = Path(r"testdata\224x224_patchs")
# directory = Path(SQUARE_PATCHS_DIR)
tif_files = list(directory.glob(f"{"roujol"}*.tif")) + list(directory.glob(f"{"gode"}*.tif"))
# tif_files = list(directory.glob("*.tif")) 
results = {}

for vec in all_vectors:
    loss_tmp = 0
    invalid = np.array([1,0,0,0,0], dtype=np.float32)  

    for tif_file in tif_files:
        _, loss, _= infer_and_display(tif_file, vec, invalid)
        loss_tmp += loss
    results[tuple(vec)] = loss_tmp / len(tif_files)
 
print("RMSE per masked channel: ", results)

import matplotlib.pyplot as plt

# Noms des canaux dans l'ordre
channel_names = ['B', 'V', 'R', 'RE', 'NIR']

# Génère les étiquettes des combinaisons sous forme de noms de canaux masqués
def vec_to_channel_label(vec):
    return ', '.join([name for i, name in enumerate(channel_names) if vec[i] == 1]) or 'None'

# Génère les labels à partir des vecteurs
comb_labels = [vec_to_channel_label(vec) for vec in results.keys()]
rmse_values = list(results.values())

# Tri optionnel par ordre alphabétique des labels
# comb_labels, rmse_values = zip(*sorted(zip(comb_labels, rmse_values), key=lambda x: x[0]))

# Tracé
plt.figure(figsize=(14, 6))
plt.bar(comb_labels, rmse_values, color='mediumseagreen')
plt.xticks(rotation=90)
plt.xlabel("Canaux masqués")
plt.ylabel("RMSE moyen")
plt.title("RMSE moyen par combinaison de canaux masqués")
plt.tight_layout()
plt.grid(axis='y')
plt.show()


def compute_average_histogram(tif_paths, bins=100):
    """
    Calcule l'histogramme moyen des valeurs de pixels pour chaque canal.
    
    :param tif_paths: Liste de chemins vers des fichiers .tif
    :param bins: Nombre de bins dans l'histogramme
    :return: Tuple (hist_moyens, bin_edges)
    """
    hist_sum = None
    for path in tif_paths:
        img = tifffile.imread(path)  # (H, W, C) ou (C, H, W)
        
        # S'assurer que l'image est au format (C, H, W)
        if img.shape[-1] == 5:
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) → (C, H, W)
        
        C, H, W = img.shape
        img = img.astype(np.float32) / 255.0  # Normalisation (si uint8)

        hists = []
        for c in range(C):
            hist, bin_edges = np.histogram(img[c].flatten(), bins=bins, range=(0, 1))
            hists.append(hist)
        
        hists = np.array(hists)  # (C, bins)
        if hist_sum is None:
            hist_sum = hists
        else:
            hist_sum += hists

    hist_moyens = hist_sum / len(tif_paths)
    return hist_moyens, bin_edges

def plot_average_histogram(hist_moyens, bin_edges, channel_names=None):
    """
    Affiche les histogrammes moyens pour chaque canal.
    
    :param hist_moyens: Tableau (C, bins)
    :param bin_edges: Bords des bins
    :param channel_names: Noms optionnels des canaux
    """
    plt.figure(figsize=(10, 6))
    C = hist_moyens.shape[0]
    colors = ['blue', 'green', 'red', 'purple', 'black']
    channel_names = channel_names or [f'Canal {i}' for i in range(C)]

    for c in range(C):
        plt.plot(bin_edges[:-1], hist_moyens[c], label=channel_names[c], color=colors[c % len(colors)])

    plt.xlabel("Valeurs de pixel (normalisées)")
    plt.ylabel("Nombre moyen de pixels")
    plt.title("Histogramme moyen par canal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# # Génération
# hist_moyens, bin_edges = compute_average_histogram(tif_files)
# plot_average_histogram(hist_moyens, bin_edges, channel_names=["B", "V", "R", "RE", "NIR"])
