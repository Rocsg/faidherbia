import random
from pathlib import Path
import tifffile
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class TifDataset(Dataset):
    """Dataset pour charger les fichiers .tif avec contrôle par nom et masquage spécifique."""
    def __init__(self, data_dir, 
                 transform=None, 
                 max_andrano = 10000):
        
        self.data_dir = Path(data_dir)
        all_tifs = list(self.data_dir.glob("*.tif"))
        
        # Séparer les .tif par type
        self.tif_files = []
        andrano_count = 0
        
        for tif in all_tifs:
            name = tif.name.lower()
            if name.startswith("andrano"):
                if andrano_count < max_andrano:
                    self.tif_files.append(tif)
                    andrano_count += 1
            else:
                self.tif_files.append(tif)
        
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tif_files)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tif_path = self.tif_files[idx]
        image = tifffile.imread(tif_path)  # (H, W, C)
        
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # (C, H, W)
        
        if self.transform:
            image = self.transform(image)
        
        mask_vector = self.create_mask(tif_path.name)
        invalid_mask = self.create_invalid_mask(tif_path.name)

        return image, mask_vector, invalid_mask

    def create_invalid_mask(self, filename: str) -> torch.Tensor:
        """Renvoie un masque des canaux *inexistants* (à exclure du calcul de la perte)."""
        invalid = torch.zeros(5)
        name = filename.lower()

        if name.startswith("roujola") or name.startswith("godet"):
            invalid[0] = 1.0  # Le canal 0 est totalement absent de l’image
            
        return invalid

    def create_mask(self, filename: str) -> torch.Tensor:
        """Génère un vecteur de masque"""
        mask = torch.zeros(5)
        name = filename.lower()

        if name.startswith("roujola") or name.startswith("godet"):
            mask[0] = 1.0  # toujours masquer le canal 0
            other_channels = [1, 2, 3, 4]
            second = random.choice(other_channels)
            mask[second] = 1.0
            
        else:
            # Cas général : masquer 1 ou 2 canaux aléatoires
            num_mask = random.choice([1, 2])
            channels = random.sample(range(5), num_mask)
            for c in channels:
                mask[c] = 1.0

        return mask