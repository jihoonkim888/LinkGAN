import torch
import torch.nn as nn
import torch.optim as optim
# import torch.utils.data
from src.MLP import *


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=7, dim=20, out_channels=1):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channels, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, dim)
        self.fc5 = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels=3, dim=20, out_channels=4):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_channels, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, dim)
        self.fc5 = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, out_channels)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.out(x)
        # x = self.sigmoid(x)
        return x


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


def init_GAN(dim, noise_dim, num_variable_links, num_conditions, device, lr_G, lr_D, beta1):
    netG = Generator(
        in_channels=noise_dim + num_conditions, dim=dim, out_channels=num_variable_links
    )
    # if run_parallel:
    #     netG = torch.nn.DataParallel(netG)
    netG = netG.to(device)
    # netG.apply(weights_init)
    netD = Discriminator(
        in_channels=num_variable_links + num_conditions, dim=dim, out_channels=1
    )
    # if run_parallel:
    #     netD = torch.nn.DataParallel(netD)
    netD = netD.to(device)
    # netD.apply(weights_init)
    # Setup Adam optimizers for both G and D
    optG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    loss = torch.nn.BCELoss()
    return netG, netD, optG, optD, loss


def init_netP(dim, num_variable_links, num_conditions, device):
    netP = MLP(in_channels=num_variable_links, out_channels=num_conditions, dim=dim)
    netP = netP.to(device)
    netP.train(False)

    weights_dir = "P_100k/d_100_lr_0.001_b1_0.9"
    weights_epoch = 1999

    weights_path = f"./weights/{weights_dir}/net_{weights_epoch}.pth"
    print("netP weights_path:", weights_path)
    netP.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    loss_P = torch.nn.MSELoss()
    return netP, loss_P