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

# Définir l'architecture regressor
class ContractingResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingResUNet, self).__init__()
        # Encodeur 256 x 256 x 1 - 64 -> 128 x 128 x 128  ->  58 x 96 x 256    
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),#To 256 x 256 x 64
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),
           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#To 128 x 128 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),
           
            nn.Conv2d(128, 128, kernel_size=3, padding=1),#To 64 x 64 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),#To 32 x 32 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),#To 16 x 16 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),#To 8 x 8 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),#To 4 x 4 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),#To 4 x 4 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),#To 4 x 4 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            #nn.AvgPool2d(2),
            nn.Conv2d(64, 2, kernel_size=1),  # 1x1 convolution acts like a linear layer
        
            #
        )
    def forward(self, x):
        # Encodage
        x1 = self.encoder(x)
        #x1 = x1.view( -1)  # Flatten the tensor
        #output = self.output_layer(x1)
        # Décodage
        return x1#output

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)