import os
import random
from pathlib import Path
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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
    Entrée: image (B, 5, 224, 224) + masque étendu (B, 5, 224, 224)
    Sortie: image reconstruite (B, 5, 224, 224)
    """
    def __init__(self):
        super().__init__()
        
        # Encoder - entrée: image (5 channels) + masque étendu (5 channels) = 10 channels
        self.enc1 = DoubleConv(10, 64)
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
        # Étendre le masque pour avoir la même taille que l'image
        mask_extended = mask_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 224, 224)
        
        # Concaténer image et masque étendu
        x = torch.cat([image, mask_extended], dim=1)  # (B, 10, 224, 224)
        
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
        
        return output


class TifDataset(Dataset):
    """Dataset pour charger les fichiers .tif avec contrôle par nom et masquage spécifique."""
    def __init__(self, data_dir: str, 
                 transform=None, 
                 max_andrano: int = 10000, 
                 p_second_channel: float = 0.5):
        
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
        self.p_second_channel = p_second_channel

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
        
        return image, mask_vector

    def create_mask(self, filename: str) -> torch.Tensor:
        """Génère un vecteur de masque selon les règles nommées."""
        mask = torch.zeros(5)
        name = filename.lower()

        if name.startswith("roujola") or name.startswith("godetc1"):
            mask[0] = 1.0  # toujours masquer le canal 0
            # Avec probabilité donnée, masquer un autre canal ≠ 0
            if random.random() < self.p_second_channel:
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

def masked_rmse_loss(predicted: torch.Tensor, target: torch.Tensor, mask_vector: torch.Tensor) -> torch.Tensor:
    """
    Calcule la RMSE uniquement sur les canaux masqués (valeur 1 dans le masque)
    
    Args:
        predicted: tensor (B, 5, 224, 224) - prédictions du modèle
        target: tensor (B, 5, 224, 224) - images cibles
        mask_vector: tensor (B, 5) - masque binaire
    """
    batch_size = predicted.shape[0]
    total_loss = torch.tensor(0.0, device=predicted.device)
    valid_samples = 0
    
    for i in range(batch_size):
        masked_channels = (mask_vector[i] == 1.0).nonzero(as_tuple=True)[0]
        
        if masked_channels.numel() > 0:
            pred_masked = predicted[i, masked_channels, :, :]
            target_masked = target[i, masked_channels, :, :]
            mse = F.mse_loss(pred_masked, target_masked, reduction='mean')
            rmse = torch.sqrt(mse)
            
            total_loss += rmse
            valid_samples += 1
    
    if valid_samples > 0:
        return total_loss / valid_samples
    else:
        return torch.tensor(0.0, device=predicted.device)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 100,
                lr: float = 1e-3,
                device: str = 'cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_val_loss = float('inf')
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Créer un répertoire pour sauvegarder les images reconstruites
    output_dir = "reconstructed_images"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Phase d'entraînement
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (images, mask_vectors) in enumerate(train_loader):
            images = images.to(device)
            mask_vectors = mask_vectors.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, mask_vectors)

            # Calcul de la loss
            loss = masked_rmse_loss(outputs, images, mask_vectors)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # Moyenne de la loss d'entraînement sur l'époque
        avg_train_loss = train_loss_accum / len(train_loader)
        train_losses.append(avg_train_loss)

        # Phase de validation
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for images, mask_vectors in val_loader:
                images = images.to(device)
                mask_vectors = mask_vectors.to(device)

                outputs = model(images, mask_vectors)
                loss = masked_rmse_loss(outputs, images, mask_vectors)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss:   {avg_val_loss:.4f}')

        # Scheduler et sauvegarde du meilleur modèle
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_channel_reconstruction_model.pth')
            print(f'  New best model saved! Val Loss: {best_val_loss:.4f}')

        # Sauvegarder une image reconstruite à la fin de chaque époque
        if len(val_loader) > 0:
            sample_images, sample_mask_vectors = next(iter(val_loader))
            sample_images = sample_images.to(device)
            sample_mask_vectors = sample_mask_vectors.to(device)

            with torch.no_grad():
                sample_outputs = model(sample_images, sample_mask_vectors)

            # Sauvegarder chaque image du batch
            for i in range(sample_images.size(0)):
                original_image = sample_images[i].cpu().numpy()          # (5, 224, 224)
                reconstructed_image = sample_outputs[i].cpu().numpy()    # (5, 224, 224)
                mask_vector = sample_mask_vectors[i].cpu().numpy()       # (5,)

                # Appliquer le masque correctement
                mask = mask_vector[:, np.newaxis, np.newaxis]  # (5, 1, 1)
                final_image = np.where(mask == 1, reconstructed_image, original_image)

                # Transposer pour (224, 224, 5)
                # final_image = np.transpose(final_image, (1, 2, 0))

                # Sauvegarder l'image
                image_path = os.path.join(output_dir, f"epoch_{epoch+1}_sample_{i}.tif")
                tifffile.imwrite(image_path, final_image)

        print('-' * 50)

    return train_losses, val_losses


def plot_training_history(train_losses: list[float], val_losses: list[float]):
    """Sauvegarde les courbes de loss d'entraînement et de validation"""
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title("Évolution des Losses pendant l'entraînement")
    plt.xlabel("Époque")
    plt.ylabel("RMSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Courbes de loss sauvegardées dans 'training_curves.png'")


if __name__ == "__main__":
    # Configuration
    data_dir = r"data\224x224_patchs"
    batch_size = 1
    num_epochs = 3
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Hyperparams → Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    print(f"Entraînement sur {device}")

    # Créer les datasets
    train_dataset = TifDataset(data_dir)
    val_dataset = TifDataset(data_dir)  # À remplacer par un dataset de validation distinct si disponible

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Créer le modèle
    model = ChannelReconstructionUNet()
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")

    # Entraîner le modèle
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Sauvegarder le plot des losses à la fin
    plot_training_history(train_losses, val_losses)
