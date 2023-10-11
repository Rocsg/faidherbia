import torch
import torch.nn as nn

# Définir une couche de bloc de résidu
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x

# Définir l'architecture ResUNet
class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()
        # Encodeur 230 x 386 x 6 - 64 -> 115 x 193 x 128  ->  58 x 96 x 256    
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),
           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),
           
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
        )
        
    def forward(self, x):
        # Encodage
        x1 = self.encoder(x)
        
        # Fusion avec des variables supplémentaires (par exemple, patch_id)
        #x1 = torch.cat((x1, patch_id), dim=1)

        # Décodage
        x2 = self.decoder(x1)

        return x2


