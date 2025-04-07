#Testing alignment with cv2
import cv2
import numpy as np

# Load the image
image_3_12 = cv2.imread('/home/rfernandez/Bureau/A_Test/Test_Sergio/Data_4/Ndvi/2024_3_12_Andrano_8bit.tif')
image_2_27 = cv2.imread('/home/rfernandez/Bureau/A_Test/Test_Sergio/Data_4/Ndvi/2024_2_27_Andrano_8bit.tif')

#Verify i can access the data by computing the mean of a square in the center
print(np.mean(image_3_12[200:300,200:300]))
print(np.mean(image_2_27[200:300,200:300]))
image_3_12 = image_3_12.astype(np.float32)[:,:,0]
image_2_27 = image_2_27.astype(np.float32)[:,:,0]
warp_matrix = np.eye(2, 3, dtype=np.float32)
print("Image 1 type :", image_3_12.dtype, "shape :", image_3_12.shape)
print("Image 2 type :", image_2_27.dtype, "shape :", image_2_27.shape)
print(image_2_27.dtype)
matrix = cv2.findTransformECC(image_3_12, image_2_27, warp_matrix, cv2.MOTION_AFFINE)
aligned = cv2.warpAffine(image_2_27, matrix, np.shape(image_2_27))