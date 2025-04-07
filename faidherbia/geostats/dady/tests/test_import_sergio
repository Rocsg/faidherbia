import tifffile as tiff
import numpy as np
import os
def downsample_tiff(input_path, output_path, output_rgbpath,factor):
    # Read the multi-slice TIFF
    print("Starting")
    with tiff.TiffFile(input_path) as tif:
        print("Go")
        images = tif.asarray()
        print(f"Original shape: {images.shape}")

    # Downsample the images
    print("Go2")
    downsampled_images = images[::factor, ::factor,:]



    blue=downsampled_images[:,:,0]
    green=downsampled_images[:,:,1]
    red=downsampled_images[:,:,2]
    infrared=downsampled_images[:,:,4]
    gray_image=np.mean([red,green,blue],axis=0)

    # Save the downsampled images as a new multi-slice TIFF
    print("Go3")

        # Stack the slices to create an RGB image
    rgb_image = np.stack((red, green, blue), axis=-1)
    downsampled_images = np.transpose(downsampled_images, (2, 0, 1))
    rgb_image = np.transpose(rgb_image, (2, 0, 1))

    if rgb_image.dtype != np.uint8:
        rgb_image = np.divide(rgb_image, 255).astype(np.uint8)

    # Save the RGB image
    tiff.imwrite(output_rgbpath, rgb_image, photometric='rgb')
    print(f"RGB image saved as {output_rgbpath}")

    tiff.imwrite(output_path, downsampled_images)
    print(f"Output shape: {downsampled_images.shape}")

    
def lister_fichiers(dossier):
    fichiers = []
    
    try:
        with os.scandir(dossier) as entries:
            for entry in entries:
                if entry.is_file():
                    fichiers.append(entry.name)
    except FileNotFoundError:
        print(f"Le dossier {dossier} n'existe pas.")
    except PermissionError:
        print(f"Permission refusée pour accéder au dossier {dossier}.")
    return fichiers

input_path = "/home/rfernandez/Bureau/Test_Sergio/Data_4/RawHighRes/"
output_path = "/home/rfernandez/Bureau/Test_Sergio/Data_4/Raw/"
output_rgbpath = "/home/rfernandez/Bureau/Test_Sergio/Data_4/Rgb/"
downsample_factor = 4
list_file=lister_fichiers(input_path)
for fich in list_file : 
    print(fich)
    downsample_tiff(input_path+""+fich, output_path+""+fich,output_rgbpath+""+fich,downsample_factor)
