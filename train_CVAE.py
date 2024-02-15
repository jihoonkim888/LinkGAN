import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
from src.data import import_data, get_dataloader
from src.CVAE import CVAE, loss_function

# Data Loading and Training Loop
# Replace these with your dataset's specifics
feature_size = 5  # Feature size of your 1D data
condition_dim = 2  # Dimension of your condition variable
latent_dim = 3  # Latent space dimension
batch_size = 10000
learning_rate = 1e-3
num_epochs = 100

workers = 0
data_path = "./data/data_100000.csv"
save_path = "./weights/CVAE"
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Example training loop
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        conditions = data[:, -condition_dim:]
        data = data[:, :-condition_dim]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, conditions)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}"
            )

    print(
        f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}"
    )
    save_model(model, "final", save_path)


def save_model(model, epoch, save_path):
    model_file = f"cvae_epoch_{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, model_file))
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    print("Importing real data...")
    input_data = import_data(data_path)
    input_tensors = torch.from_numpy(input_data).float()
    print("input_data shape:", input_data.shape)

    print("Initializing dataloader...")
    train_loader = get_dataloader(input_tensors, batch_size, workers)
    # Initialize the CVAE model
    print("Initializing CVAE model...")
    cvae = CVAE(
        input_dim=feature_size, condition_dim=condition_dim, latent_dim=latent_dim
    )
    optimizer = Adam(cvae.parameters(), lr=learning_rate)

    # Assuming train_loader is defined and loaded with your 1D data
    print("Starting model training...")
    for epoch in range(1, num_epochs + 1):
        train(cvae, train_loader, optimizer, epoch)
