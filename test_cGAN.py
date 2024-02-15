import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from src.data import import_data
from src.loss_functions import diversity_loss
from src.GAN import Generator, init_netP
from src.NearestNeighbors import generate_synthetic_data
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_samples", type=int, default=100, help="Number of samples to generate"
)
parser.add_argument("--dp", type=str, required=True)
parser.add_argument("--wp", type=str, required=True, help="weight path of Generator")
args = parser.parse_args()

print("weight path:", args.wp)

# PARAMETERS
num_conditions = 2  # d_max, ratio_min
num_variable_links = 5  # l2, l3, l4, EE_x, EE_y

dim = 100
noise_dim = 5
k = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("device:", device)


def init_netG():
    netG = Generator(
        in_channels=noise_dim + num_conditions, dim=dim, out_channels=num_variable_links
    )
    netG = netG.to(device)
    return netG


csv_save_dir = "./synthesis_100k"
os.makedirs(csv_save_dir, exist_ok=True)

# Get the current date and time
current_datetime = datetime.now()
# Format the date and time as per the specified format: 240124_1939
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")
filename = f"synthesis_n_{args.num_samples}_{formatted_datetime}.csv"
csv_path = os.path.join(csv_save_dir, filename)


if __name__ == "__main__":
    print("Initialising Generator...")
    netG = init_netG()
    netG.load_state_dict(torch.load(args.wp, map_location=torch.device(device)))
    netP, loss_P = init_netP(dim, num_variable_links, num_conditions, device)

    print(f"Generating {args.num_samples} samples...")
    input_data = import_data(args.dp)
    arr_d_max, arr_eta_min = generate_synthetic_data(
        input_data[:, -2:], args.num_samples, k
    )
    conds = torch.cat((arr_d_max, arr_eta_min), 1)

    # input to G
    noise = torch.randn(args.num_samples, noise_dim, device=device)
    netG_input = torch.cat((noise, arr_d_max, arr_eta_min), 1)

    # output from G
    with torch.no_grad():
        fake = netG(netG_input)
        conds_P = netP(fake[:1000])
        l_P = loss_P(conds_P, conds[:1000])
        l_div = -diversity_loss(fake[:1000])
    print("1000 samples l_P:", l_P, "l_div:", l_div)

    output = torch.cat((fake, arr_d_max, arr_eta_min), 1)

    output = output.detach().cpu().numpy()
    print(output.shape)
    cols = ["l2", "l3", "l4", "EE_x", "EE_y", "d_max", "eta_min"]
    df = pd.DataFrame(output, columns=cols)

    df.to_csv(csv_path, index=False)
    print(f"Done! The synthesised samples are saved to {csv_path}")
