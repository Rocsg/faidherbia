import os
import shutil
import random
import pandas as pd



# Set the paths for the source and destination directories
source_data_dir = "/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/Full_dataset"
train_valid_dir = "/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/TrainValid_dataset"
test_dir        = "/home/rfernandez/Bureau/A_Test/Mansour_Sustain_Sahel/DeepEstimator/Data/2021/Test_dataset"

# Create train/validate and test directories if they don't exist
os.makedirs(train_valid_dir, exist_ok=True)
os.makedirs(train_valid_dir+'/MultiSpectralImgs', exist_ok=True)
os.makedirs(train_valid_dir+'/TargetValues', exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(test_dir+'/MultiSpectralImgs', exist_ok=True)
os.makedirs(test_dir+'/TargetValues', exist_ok=True)

# Specify the path to the single CSV file
csv_file_path = os.path.join(source_data_dir, "TargetValues/Data_faidherbia.csv")

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Split ratio for train/validate and test datasets (adjust as needed)
train_valid_ratio = 0.7  # 70% for train/validate, 30% for test

# Randomly shuffle the DataFrame to ensure a random split
df = df.sample(frac=1, random_state=42)

# Calculate the split indices
split_index = int(len(df) * train_valid_ratio)

# Split the DataFrame into train/validate and test sets
train_valid_df = df[:split_index]
test_df = df[split_index:]

# Save the filtered CSV files for train/validate and test datasets
train_valid_csv_path = os.path.join(train_valid_dir, "Data_faidherbia.csv")
test_csv_path = os.path.join(test_dir, "Data_faidherbia.csv")

train_valid_df.to_csv(train_valid_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# Copy the corresponding images to TrainValid_dataset/MultiSpectralImgs and Test_dataset/MultiSpectralImgs
for image_identifier in train_valid_df['#ID-Placette']:
    image_file = image_identifier + ".tif"  # Assuming images are in jpg format
    src_image = os.path.join(source_data_dir, "MultiSpectralImgs", image_file)
    dst_image = os.path.join(train_valid_dir, "MultiSpectralImgs", image_file)
    print("Copying {} to {}".format(src_image, dst_image)   )
    shutil.copy2(src_image, dst_image)

for image_identifier in test_df['#ID-Placette']:
    image_file = image_identifier + ".tif"  # Assuming images are in jpg format
    src_image = os.path.join(source_data_dir, "MultiSpectralImgs", image_file)
    dst_image = os.path.join(test_dir, "MultiSpectralImgs", image_file)
    shutil.copy2(src_image, dst_image)

print("Data splitting and copying completed.")
