from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from torch.utils.tensorboard import SummaryWriter


dady_root = Path(__file__).parent.parent  # tests/ -> DADY/
sys.path.insert(0, str(dady_root))

from utils.config_utils.path_utils import SQUARE_PATCHS_DIR
from models_archi.unet.unet_model import ChannelReconstructionUNet, masked_rmse_loss
from models_archi.unet.unet_data_loader import TifDataset



def train_model(model, train_loader, val_loader, num_epochs=3, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    writer = SummaryWriter(log_dir='../tensorboard_runs/channel_reconstruction/v2')

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (images, mask_vectors, invalid_masks) in enumerate(train_loader):
            images = images.to(device)
            mask_vectors = mask_vectors.to(device)
            invalid_masks = invalid_masks.to(device)

            optimizer.zero_grad()
            outputs = model(images, mask_vectors)
            loss = masked_rmse_loss(outputs, images, mask_vectors, invalid_masks)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

            # Log training loss every 10 batches
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), global_step)
        
        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation phase
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for images, mask_vectors, invalid_masks in val_loader:
                images = images.to(device)
                mask_vectors = mask_vectors.to(device)
                invalid_masks = invalid_masks.to(device)
                outputs = model(images, mask_vectors)
                loss = masked_rmse_loss(outputs, images, mask_vectors, invalid_masks)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}')

        # Log average losses per epoch
        writer.add_scalar('Loss/avg_train', avg_train_loss, epoch)
        writer.add_scalar('Loss/avg_val', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'reconstruction_model{epoch}.pth')
            print(f'New model saved! Val Loss: {best_val_loss:.4f}')

        # Optionally log reconstructed images to TensorBoard
        if len(val_loader) > 0:
            sample_images, sample_mask_vectors, sample_invalid_masks  = next(iter(val_loader))
            sample_images = sample_images.to(device)
            sample_mask_vectors = sample_mask_vectors.to(device)
            sample_invalid_masks = sample_invalid_masks.to(device)

            with torch.no_grad():
                sample_outputs = model(sample_images, sample_mask_vectors)

            # Log the first image of the batch
            original = sample_images[0].cpu()
            reconstructed = sample_outputs[0].cpu()

            for ch in range(5):
                writer.add_image(f'Original/channel_{ch}', original[ch, :, :].unsqueeze(0), epoch)
                writer.add_image(f'Reconstructed/channel_{ch}', reconstructed[ch, :, :].unsqueeze(0), epoch)



    writer.close()
    
from torch.utils.data import random_split, DataLoader

# Hyperparameters
data_dir = SQUARE_PATCHS_DIR  # Path to your .tif files
batch_size = 32
num_epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Hyperparams → Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
print(f"Entraînement sur {device}")

# Load full dataset
full_dataset = TifDataset(data_dir)

# Load trained model
model = ChannelReconstructionUNet()
model.load_state_dict(torch.load(r"models_saved\reconstruction_model30.pth", map_location=device))
model.to(device)

model.eval()

# Load a single batch from validation set
val_loader = DataLoader(full_dataset, batch_size=40, shuffle=True)
images, mask_vectors, invalid_masks = next(iter(val_loader))
images = images.to(device)
mask_vectors = mask_vectors.to(device)
invalid_masks = invalid_masks.to(device)

with torch.no_grad():
    outputs = model(images, mask_vectors)
    loss = masked_rmse_loss(outputs, images, mask_vectors, invalid_masks)
print(f"Loss on a single batch using loaded model: {loss.item():.4f}")


import matplotlib.pyplot as plt

# Sélection de la dernière image du batch
idx = -1  # dernière image du batch
original = images[idx].cpu()
output = outputs[idx].cpu()
mask = mask_vectors[idx].cpu()

# Création de l'image masquée (valeurs mises à zéro là où mask == 1)
masked = original.clone()
for c in range(5):
    if mask[c] == 1:
        masked[c] = 0.0

# Plot
fig, axs = plt.subplots(5, 3, figsize=(12, 10))
for i in range(5):
    axs[i, 0].imshow(original[i], cmap='gray')
    axs[i, 0].set_title(f"Canal {i} - Original")

    axs[i, 1].imshow(masked[i], cmap='gray')
    axs[i, 1].set_title(f"Canal {i} - Masqué")

    axs[i, 2].imshow(output[i], cmap='gray')
    axs[i, 2].set_title(f"Canal {i} - Reconstruit")

    for j in range(3):
        axs[i, j].axis("off")

plt.tight_layout()
plt.show()
