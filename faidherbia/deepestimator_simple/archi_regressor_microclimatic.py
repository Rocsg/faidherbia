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
           
            nn.Conv2d(128, 256, kernel_size=3, padding=1),#To 64 x 64 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 32 x 32 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 16 x 16 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 8 x 8 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 4 x 4 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 2 x 2 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.AvgPool2d(2),#To 1 x 1 x 256
        
            #
        )
        # Modify the output layer for a fully connected layer
        self.output_layer = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(256, out_channels),  # Change the number of output channels
            nn.ReLU(inplace=True),
        )
         
    def forward(self, x):
        # Encodage
        x1 = self.encoder(x)
        
        # Fusion avec des variables supplémentaires (par exemple, patch_id)
        #x1 = torch.cat((x1, patch_id), dim=1)
        output = self.output_layer(x1)
        # Décodage
        return output


# Définir l'architecture regressor
class ContractingResUNetWithMicroclimaticVariables(nn.Module):
    def __init__(self, in_channels_image,in_channels_numeric, out_channels):
        super(ContractingResUNet, self).__init__()
        # Encodeur 224 x 380 x 6 - 64 -> 112 x 190 x 64  ->  56 x 95 x 128 --> 28 x 47 x 256 --> 14 x 24 x 256     
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels_image, 64, kernel_size=3, padding=1),#To 224 x 380 x 6 - 64
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels_image, 64, kernel_size=3, padding=1),#To 112 x 190 x 64
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),
           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#To 56 x 95 x 128
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),
           
            nn.Conv2d(128, 256, kernel_size=3, padding=1),#To 28 x 47 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 14 x 24 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),#To 7 x 12 x 256
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),
 
  
       )

        # Convolutional layers for numeric variables
        self.conv_numeric = nn.Sequential(
            nn.Conv1d(in_channels_numeric, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Modify the output layer to produce 3 single values
        self.output_layer = nn.Sequential(
            nn.Conv2d(256+32, out_channels, kernel_size=1),  # Change the number of output channels
        )

        
    def forward(self, x_image, x_numeric):
        # Encoder for the image
        x_image = self.encoder(x_image)

        # Convolutional layers for numeric variables
        x_numeric = self.conv_numeric(x_numeric)

        # Concatenate image and numeric features along the channel dimension
        x_combined = torch.cat((x_image, x_numeric), dim=1)

        # Compute output
        output = self.output_layer(x_combined)

        return output