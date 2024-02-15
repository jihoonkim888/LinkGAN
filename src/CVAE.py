import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x, c):
        x = F.relu(self.fc1(torch.cat([x, c], 1)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Decoder Network
class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, z, c):
        z = F.relu(self.fc1(torch.cat([z, c], 1)))
        reconstruction = torch.sigmoid(self.fc2(z))
        return reconstruction


# Reparameterization Trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# Conditional Variational Autoencoder
class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, input_dim)

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar


# Loss Function
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
