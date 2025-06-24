import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional

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

def masked_rmse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask_vector: torch.Tensor,
    invalid_mask_vector: torch.Tensor
) -> torch.Tensor:
    """
    Calcule la RMSE uniquement sur les canaux masqués (mask_vector == 1),
    en excluant les canaux invalides (invalid_mask_vector == 1).
    
    Args:
        predicted: (B, 5, H, W) - Prédictions du modèle
        target: (B, 5, H, W) - Vérités terrain
        mask_vector: (B, 5) - Masques de canaux à reconstruire
        invalid_mask_vector: (B, 5) - Masques de canaux invalides à ignorer
    """
    batch_size = predicted.shape[0]
    total_loss = torch.tensor(0.0, device=predicted.device)
    valid_samples = 0
    
    for i in range(batch_size):
        # Masques de reconstruction ET valides
        effective_mask = (mask_vector[i] == 1.0) & (invalid_mask_vector[i] == 0.0)
        masked_channels = effective_mask.nonzero(as_tuple=True)[0]
        
        print(masked_channels)

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
