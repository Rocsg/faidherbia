from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from torch.utils.tensorboard import SummaryWriter


dady_root = Path(__file__).parent.parent.parent  # tests/ -> DADY/
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

if __name__ == "__main__":

    data_dir =  SQUARE_PATCHS_DIR # Chemin vers le dossier contenant les fichiers .tif
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Hyperparams → Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    print(f"Entraînement sur {device}")

    # Charger l'ensemble complet
    full_dataset = TifDataset(data_dir)

    # Split 80% / 20%
    val_split = 0.2
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Créer le modèle
    model = ChannelReconstructionUNet()
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")

    # Entraîner le modèle
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
