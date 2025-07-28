import tifffile
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# dady_root = Path(__file__).parent.parent  # tests/ -> DADY/
# sys.path.insert(0, str(dady_root))
# from utils.raw_archi.unet import ChannelReconstructionUNet

class TifDataset(Dataset):
    """Dataset pour charger les fichiers .tif avec contrôle par nom et masquage spécifique."""
    def __init__(self, 
                 data_dir, 
                 transform=None):
        
        self.data_dir = Path(data_dir)
        all_tifs = list(self.data_dir.glob("*.tif"))
        
        # Séparer les .tif par type
        self.tif_files = []
        
        for tif in all_tifs:
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

class DoubleConv(nn.Module):
    """Double convolution block """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class ChannelReconstructionUNet(nn.Module):
    """
    U-Net modifié pour la reconstruction de canaux manquants
    Entrée: image (B, 5, 224, 224) 
    Sortie: image reconstruite (B, 5, 224, 224)
    """
    def __init__(self):
        super().__init__()
        
        # Encoder - entrée: image (5 channels) + masque étendu (5 channels) = 10 channels
        self.enc1 = DoubleConv(5, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Couche de sortie
        self.final_conv = nn.Conv2d(64, 5, kernel_size=1)
        
    def forward(self, image: torch.Tensor, mask_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: tensor (B, 5, 224, 224) - image d'entrée
            mask_vector: tensor (B, 5) - vecteur binaire indiquant les canaux à reconstruire
        """
        # Masquer les canaux à reconstruire 
        image = image.clone()  # pour éviter de modifier l'entrée originale
        for i in range(image.shape[0]):
            for c in range(5):
                if mask_vector[i, c] == 1:
                    image[i, c] = -1.0
                    
        x = image
        
        # Encoder
        e1 = self.enc1(x)      # (B, 64, 224, 224)
        p1 = self.pool1(e1)    # (B, 64, 112, 112)
        
        e2 = self.enc2(p1)     # (B, 128, 112, 112)
        p2 = self.pool2(e2)    # (B, 128, 56, 56)
        
        e3 = self.enc3(p2)     # (B, 256, 56, 56)
        p3 = self.pool3(e3)    # (B, 256, 28, 28)
        
        e4 = self.enc4(p3)     # (B, 512, 28, 28)
        p4 = self.pool4(e4)    # (B, 512, 14, 14)
        
        # Bottleneck
        b = self.bottleneck(p4)  # (B, 1024, 14, 14)

        # Decoder avec skip connections
        up4 = self.upconv4(b)          # (B, 512, 28, 28)
        merge4 = torch.cat([up4, e4], dim=1)  # (B, 1024, 28, 28)
        d4 = self.dec4(merge4)         # (B, 512, 28, 28)
        
        up3 = self.upconv3(d4)         # (B, 256, 56, 56)
        merge3 = torch.cat([up3, e3], dim=1)  # (B, 512, 56, 56)
        d3 = self.dec3(merge3)         # (B, 256, 56, 56)
        
        up2 = self.upconv2(d3)         # (B, 128, 112, 112)
        merge2 = torch.cat([up2, e2], dim=1)  # (B, 256, 112, 112)
        d2 = self.dec2(merge2)         # (B, 128, 112, 112)
        
        up1 = self.upconv1(d2)         # (B, 64, 224, 224)
        merge1 = torch.cat([up1, e1], dim=1)  # (B, 128, 224, 224)
        d1 = self.dec1(merge1)         # (B, 64, 224, 224)
        
        # Sortie finale
        output = self.final_conv(d1)   # (B, 5, 224, 224)
        output = torch.sigmoid(output)
        
        return output

def masked_rmse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask_vector: torch.Tensor,
    multiplier: float = 2.0,
    validation: bool = False
) -> torch.Tensor:
    """
    Calcule le RMSE pondéré selon que les canaux sont masqués ou non.

    Args:
        predicted: (B, C, H, W)
        target:    (B, C, H, W)
        mask_vector: (B, C) - 1 = canal masqué, 0 = canal normal
        multiplier: facteur de pondération pour canaux masqués
        validation: si True, désactive la pondération

    Returns:
        Scalaire : RMSE moyen pondéré sur le batch et les canaux
    """
    # (B, C, H, W)
    se = F.mse_loss(predicted, target, reduction='none')

    # Moyenne spatiale : reste (B, C)
    mse_per_channel = se.mean(dim=(2, 3))

    # Racine carrée pour RMSE
    rmse_per_channel = torch.sqrt(mse_per_channel + 1e-8)

    # Pondération
    if validation:
        weights = torch.ones_like(mask_vector, dtype=rmse_per_channel.dtype)
    else:
        weights = torch.where(mask_vector == 1,
                              torch.tensor(multiplier, dtype=rmse_per_channel.dtype, device=rmse_per_channel.device),
                              torch.tensor(1.0, dtype=rmse_per_channel.dtype, device=rmse_per_channel.device))

    weighted_rmse = rmse_per_channel * weights

    # Moyenne batch et canaux
    loss = weighted_rmse.mean()

    return loss





def train_model(model, train_loader, val_loader,
                optimizer, scheduler,
                num_epochs=3, device='cuda', start_epoch=0):
    model = model.to(device)
    
    writer = SummaryWriter(log_dir='../tensorboard_runs/channel_reconstruction/x2ponderation')

    best_val_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        train_epoch_loss = 0.0
        tensorboard_loss = 0.0

        for batch_idx, (images, mask_vectors, invalid_masks) in enumerate(train_loader):
            images = images.to(device)
            mask_vectors = mask_vectors.to(device)
            invalid_masks = invalid_masks.to(device)


            optimizer.zero_grad()
            outputs = model(images, mask_vectors)
            loss = masked_rmse_loss(outputs, images, invalid_masks, validation=False)
            loss.backward()
            optimizer.step()
            tensorboard_loss += masked_rmse_loss(outputs, images, invalid_masks, validation=True).item()
            train_epoch_loss += loss.item()

            # Log training loss every 10 batches
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), global_step)
        
        avg_train_loss = tensorboard_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for images, mask_vectors, invalid_masks in val_loader:
                images = images.to(device)
                mask_vectors = mask_vectors.to(device)
                invalid_masks = invalid_masks.to(device)
                outputs = model(images, mask_vectors)
                loss = masked_rmse_loss(outputs, images, invalid_masks, validation=True)
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}')

        # Log average losses per epoch
        writer.add_scalar('Loss/avg_train', avg_train_loss, epoch)
        writer.add_scalar('Loss/avg_val', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss or epoch % 10 == 0:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch':      epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss':        best_val_loss,
            }, f'x2ponderation_{epoch}.pth')
            print(f'Checkpoint saved! Val Loss: {best_val_loss:.4f}')


        # Optionally log reconstructed images to TensorBoard
        if len(val_loader) > 0:
            sample_images, sample_mask_vectors, sample_invalid_masks  = next(iter(val_loader))
            sample_images = sample_images.to(device)
            sample_mask_vectors = sample_mask_vectors.to(device)
            sample_invalid_masks = sample_invalid_masks.to(device)

            with torch.no_grad():
                sample_outputs = model(sample_images, sample_mask_vectors)

            # Log the first image of the batch
            original = sample_images[0].cpu()
            reconstructed = sample_outputs[0].cpu()

            for ch in range(5):
                writer.add_image(f'Original/channel_{ch}', original[ch, :, :].unsqueeze(0), epoch)
                writer.add_image(f'Reconstructed/channel_{ch}', reconstructed[ch, :, :].unsqueeze(0), epoch)



# data_dir =  r"testdata\224x224_patchs" # Chemin vers le dossier contenant les fichiers .tif

batch_size = 32
num_epochs = 200
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hyperparams → Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
print(f"Entraînement sur {device}")

train_dataset = TifDataset(r"/lustre/fswork/projects/rech/xfz/uuh33xb/data/224x224_train")
val_dataset = TifDataset(r"/lustre/fswork/projects/rech/xfz/uuh33xb/data/224x224_test")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Créer le modèle
model = ChannelReconstructionUNet()
print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
# Entraîner le modèle
train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
