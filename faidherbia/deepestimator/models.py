import torch
import torch.nn as nn
import torchvision.models as models
from faidherbia.datahandling.utils import nbPlacettes

nb_microclimatic_features=4+nbPlacettes()
num_feat_image=2
num_feat_intermediate=32
# Supprimer la couche de classification par défaut




def get_arch(num_feat_image,features_extractor_arch='densenet',grayscale=False):
    if(features_extractor_arch=='densenet'):
        model = models.densenet201(pretrained=True)
        if(grayscale): 
            model.features.conv0 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(1920, num_feat_image)  # Apply sigmoid activation for each output
        )
    else:
        model= models.resnet50(pretrained=True)
        if(grayscale): 
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
           nn.Linear(num_features, 64),
#            nn.ReLU(),
#            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_feat_image)
        )

    

    return model



# Créer un modèle personnalisé qui fusionne les caractéristiques extraites et les variables numériques
class MyModel(nn.Module):
    def __init__(self, num_feat_image, use_microclimatic,features_extractor_arch='densenet',grayscale=False):
        super(MyModel, self).__init__()
        #Build arch of features extractor
        self.use_microclimatic=use_microclimatic
        if(not use_microclimatic):
           num_feat_image=2
        self.num_feat_image = num_feat_image
        self.features_extractor=get_arch(num_feat_image,features_extractor_arch,grayscale=grayscale)
        self.numeric_layer = nn.Sequential(
            nn.Linear(nb_microclimatic_features, num_feat_intermediate),  # Couche pour les variables numériques
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_feat_intermediate + num_feat_image, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2)
        )

    def forward(self, image, numeric_features=None):
        image_features = self.features_extractor(image)
        if(self.use_microclimatic):
            numeric_features = self.numeric_layer(numeric_features)
            combined_features = torch.cat((image_features, numeric_features), dim=1)
            output = self.classifier(combined_features)
            return output
        else:
            return image_features

# Exemple d'utilisation du modèle
image = torch.randn(1, 3, 224, 224)  # Exemple d'image (batch_size, channels, height, width)
numeric_input = torch.randn(1, 5)  # Exemple de tenseur de variables numériques (batch_size, num_numeric_features)

custom_model = MyModel(num_feat_image=32, use_microclimatic=True)
