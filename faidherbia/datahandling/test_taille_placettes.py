import os
import random
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
data_dir = '/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/TrainValid_dataset/MultiSpectralImgs'

#Lister les images presentes dans datadir
image_list = os.listdir(data_dir)
surf=[]
X=[]
Y=[]
for img in image_list:
    img_name = os.path.join(data_dir, img)
    image_input = io.imread(img_name)
#    print(np.shape(image_input))
    img_ir=image_input[:,:,4]
    img_r=image_input[:,:,2]
    img_ndvi=(img_ir-img_r)/(img_ir+img_r)
    #print(np.shape(img_ndvi))

 #   x = random.randint(0, np.shape(img_ndvi)[0]-256)
 #   y = random.randint(0, np.shape(img_ndvi)[1]-256)
 #   img_ndvi = img_ndvi[x:x+256,y:y+256]   
        
    surface=np.shape(img_ndvi)[0]*np.shape(img_ndvi)[1]

    #append surface to surf
    surf.append(surface)
    X.append(np.shape(img_ndvi)[0])
    Y.append(np.shape(img_ndvi)[1])
print("-----------------")
print("-----------------")
print("Surface des placettes")
print(surf)
#Afficher des statistiques rudimentaires sur surf : min, max, moyenne, mediane, ecart type
print("min : "+str(min(surf)))
print("max : "+str(max(surf)))
print("mean : "+str(np.mean(surf)))
print("median : "+str(np.median(surf)))
print("std : "+str(np.std(surf)))

print("-----------------")
print("-----------------")
print("Hauteur des placettes")
print(str(X))
#Afficher des statistiques rudimentaires sur X : min, max, moyenne, mediane, ecart type
print("min : "+str(min(X)))
print("max : "+str(max(X)))
print("mean : "+str(np.mean(X)))
print("median : "+str(np.median(X)))
print("std : "+str(np.std(X)))

print("-----------------")
print("-----------------")
print("Largeur des placettes")
print(str(Y))
#Afficher des statistiques rudimentaires sur Y : min, max, moyenne, mediane, ecart type
print("min : "+str(min(Y)))
print("max : "+str(max(Y)))
print("mean : "+str(np.mean(Y)))
print("median : "+str(np.median(Y)))
print("std : "+str(np.std(Y)))




print()