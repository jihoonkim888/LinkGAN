import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random
import torch
from src.data import import_data, get_dataloader
from src.GAN import init_GAN, init_netP
from src.loss_functions import diversity_loss
from src.plot import plot_convergence, plot_loss_P_Var
from src.NearestNeighbors import generate_synthetic_data
import argparse


# PARAMETERS
num_conditions = 2  # d_max, ratio_min
num_variable_links = 5  # l2, l3 and l4, EE_x, EE_y

# PARAMETERS FROM ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument("-e", "--num_epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=10000)
parser.add_argument("-d", "--dim", type=int, default=100)
parser.add_argument(
    "-nd", "--noise_dim", type=int, required=False
)  # noise dim for generator
parser.add_argument(
    "-dp", "--data_path", type=str, required=True
)  # path where the input csv file is saved
parser.add_argument(
    "-wp", "--weight_path", type=str, required=True
)  # path to save network weights
parser.add_argument("-b1", "--beta1", type=float, required=False)
parser.add_argument("-lrg", "--learning_rate_G", type=float, required=True)
parser.add_argument("-lrd", "--learning_rate_D", type=float, required=True)
parser.add_argument("-pw", "--weight_predictor", type=float, required=True)
parser.add_argument("-dw", "--weight_diversity", type=float, required=True)

args = parser.parse_args()


# argparse
dim = args.dim  # dimension of the each MLP layer in G and D
num_epochs = args.num_epochs
batch_size = args.batch_size
data_path = args.data_path
weights_path = args.weight_path
lr_G = args.learning_rate_G
lr_D = args.learning_rate_D
beta1 = args.beta1 if args.beta1 else 0.9
noise_dim = args.noise_dim if args.noise_dim else 5  # latent space vector dim
w_P = args.weight_predictor  # weight (lambda_P) for MSE loss from netP
w_div = args.weight_diversity


k = 5  # k for SMOTE algorithm


workers = 0
# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("device:", device)

param_info = f"d_{dim}_nd_{noise_dim}_b_{batch_size}_lrg_{lr_G}_lrd_{lr_D}_b1_{beta1}_pw_{w_P}_dw_{w_div}"
weights_path = os.path.join(weights_path, param_info)
os.makedirs(weights_path, exist_ok=True)


def save_weights(epoch):
    netG_filename = os.path.join(weights_path, f"netG_e{epoch}.pth")
    netD_filename = os.path.join(weights_path, f"netD_e{epoch}.pth")
    torch.save(netG.state_dict(), netG_filename)
    torch.save(netD.state_dict(), netD_filename)


def quality_score(arr_d_eval, arr_gr_min_eval):
    netG_input = torch.cat(
        (noise_eval, arr_d_eval, arr_gr_min_eval), 1
    )  # stack vertically
    with torch.no_grad():  # do not update netP
        fake = netG(netG_input).detach()
        conds_P = netP(fake)
        score = float(loss_P(conds_P, conds_eval))
    return score


def diversity_score():
    """The mean of the closest pairwise Euclidean distances between samples"""
    netG_input = torch.cat(
        (noise_eval, arr_d_eval, arr_gr_min_eval), 1
    )  # stack vertically
    with torch.no_grad():
        fake = netG(netG_input).detach()
        score = float(-diversity_loss(fake))
    return score


def run_GAN_P_div(dataloader, netG, netD, netP, optG, optD, loss, loss_P):
    # Lists to keep track of progress
    G_losses = []
    D_real_losses = []
    D_fake_losses = []
    P_losses = []
    Div_losses = []
    real_accuracies = []
    fake_accuracies = []
    lst_results = []

    ##### START OF EPOCH #####
    for epoch in tqdm(range(num_epochs)):
        # append average of errors and accuracies after every epoch
        lst_errD_real_batch = []
        lst_errD_fake_batch = []
        lst_errG_batch = []
        lst_errP_batch = []
        lst_errDiv_batch = []
        lst_train_acc_real_batch = []
        lst_train_acc_fake_batch = []
        lst_update = []

        ### START OF BATCH ###
        for data in dataloader:
            ### START OF DISCRIMINATOR UPDATE ###
            optD.zero_grad()
            # Discriminator on real data #
            real_data = data.to(device)
            label_real = torch.full(
                (batch_size,), 1.0, dtype=torch.float, device=device
            )

            outD_real = netD(real_data).view(-1)
            errD_real = loss(outD_real, label_real)
            errD_real.backward()
            lst_errD_real_batch.append(errD_real.item())

            # D acc for real samples
            train_acc_real = (
                torch.sum((outD_real > 0.5).to(int) == label_real) / batch_size
            ).item()
            lst_train_acc_real_batch.append(train_acc_real)

            # Update Discriminator with fake data generated from noise #
            label_fake = torch.full(
                (batch_size,), 0.0, dtype=torch.float, device=device
            )
            noise = torch.randn(batch_size, noise_dim, device=device)
            # noise = torch.rand(batch_size, noise_dim, device=device)

            # generate conditions (delta_x, delta_y, gear_ratio_min)
            arr_d, arr_gr_min = generate_synthetic_data(
                input_data[:, -2:], batch_size, k
            )
            arr_d = arr_d.to(device)
            arr_gr_min = arr_gr_min.to(device)
            netG_input = torch.cat((noise, arr_d, arr_gr_min), 1)  # stack vertically
            fake = netG(netG_input).detach()
            netD_input_fake = torch.cat((fake, arr_d, arr_gr_min), 1)
            outD_fake = netD(netD_input_fake).view(-1)
            errD_fake = loss(outD_fake, label_fake)
            errD_fake.backward()
            lst_errD_fake_batch.append(errD_fake.item())

            # D acc for samples from G
            train_acc_fake = (
                torch.sum((outD_fake > 0.5).to(int) == label_fake) / batch_size
            ).item()
            lst_train_acc_fake_batch.append(train_acc_fake)

            acc_real_mean = np.mean(lst_train_acc_real_batch)
            acc_fake_mean = np.mean(lst_train_acc_fake_batch)
            update = ((acc_real_mean + acc_fake_mean) / 2) < 0.8
            lst_update.append(update)
            if update:
                optD.step()
            optD.zero_grad()
            ### END OF DISCRIMINATOR UPDATE ###

            ### START OF GENERATOR UPDATE ###
            optG.zero_grad()
            label = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            # noise = torch.rand(batch_size, noise_dim, device=device)
            noise = torch.randn(batch_size, noise_dim, device=device)
            arr_d, arr_gr_min = generate_synthetic_data(
                input_data[:, -2:], batch_size, k
            )
            arr_d = arr_d.to(device)
            arr_gr_min = arr_gr_min.to(device)
            conds = torch.cat((arr_d, arr_gr_min), 1)
            netG_input = torch.cat((noise, arr_d, arr_gr_min), 1)  # stack vertically
            fake = netG(netG_input)  # fake link sets
            with torch.no_grad():  # do not update netP
                conds_P = netP(fake)
            netD_input_fake = torch.cat((fake, arr_d, arr_gr_min), 1)
            output = netD(netD_input_fake).view(-1)

            lG = loss(output, label)
            lP = loss_P(conds_P, conds)
            # lVar = VarLoss(fake)
            lDiv = diversity_loss(fake)

            errG = lG + lP * w_P + lDiv * w_div
            errG.backward()
            lst_errG_batch.append(lG.item())
            lst_errP_batch.append(lP.item())
            lst_errDiv_batch.append(lDiv.item())

            optG.step()
            optG.zero_grad()
            ### END OF GENERATOR UPDATE ###

        ##### START OF EPOCH #####
        G_losses.append(np.sum(lst_errG_batch))
        D_real_losses.append(np.sum(lst_errD_real_batch))
        D_fake_losses.append(np.sum(lst_errD_fake_batch))
        P_losses.append(np.sum(lst_errP_batch))
        Div_losses.append(np.sum(lst_errDiv_batch))
        real_accuracies.append(np.mean(lst_train_acc_real_batch))
        fake_accuracies.append(np.mean(lst_train_acc_fake_batch))
        #### END OF EPOCH #####

        # save net weights every 10 epochs
        if (epoch % 5 == 0 and epoch != 0) or (epoch == num_epochs - 1):
            # save network weights
            netG_filename = f"{weights_path}/netG_{epoch}.pth"
            netD_filename = f"{weights_path}/netD_{epoch}.pth"
            torch.save(netG.state_dict(), netG_filename)
            torch.save(netD.state_dict(), netD_filename)

            # evaluate the model and record the results
            results = {
                "lr_G": lr_G,
                "lr_D": lr_D,
                "w_P": w_P,
                "w_div": w_div,
                "epoch": epoch,
                "quality_score": quality_score(arr_d_eval, arr_gr_min_eval),
                "diversity_score": diversity_score(),
            }
            print(results)
            lst_results.append(results)

        # if (epoch % 10 == 0) and (epoch != 0):
        plot_convergence(
            G_losses,
            D_real_losses,
            D_fake_losses,
            real_accuracies,
            fake_accuracies,
            lr_G,
            lr_D,
            weights_path,
        )
        plot_loss_P_Var(P_losses, Div_losses, w_P, w_div, weights_path)
        #### END OF EPOCH #####

    return lst_results


if __name__ == "__main__":
    t0 = time.perf_counter()

    # make sure not to re-run with the same hyperparameters
    assert not os.path.exists(
        os.path.join(weights_path, "results.csv")
    ), "results.csv already exists!"

    netG, netD, optG, optD, loss = init_GAN(
        dim, noise_dim, num_variable_links, num_conditions, device, lr_G, lr_D, beta1
    )
    netP, loss_P = init_netP(dim, num_variable_links, num_conditions, device)

    print("Importing real data...")
    input_data = import_data(data_path)
    input_tensors = torch.from_numpy(input_data).float()
    print("input_data shape:", input_data.shape)
    print("Done!")

    # conditions for model evaluation
    noise_eval = torch.randn(batch_size, noise_dim, device=device)
    arr_d_eval, arr_gr_min_eval = generate_synthetic_data(
        input_data[:, -2:], batch_size, k
    )
    arr_d_eval = arr_d_eval.to(device)
    arr_gr_min_eval = arr_gr_min_eval.to(device)
    conds_eval = torch.cat((arr_d_eval, arr_gr_min_eval), 1)

    print("Initialising dataloader...")
    dataloader = get_dataloader(input_tensors, batch_size, workers)
    print("Done!")

    print("Starting Training Loop...")
    lst_results = run_GAN_P_div(dataloader, netG, netD, netP, optG, optD, loss, loss_P)
    pd.DataFrame(lst_results).to_csv(
        os.path.join(weights_path, "results.csv"), index=False
    )

    t1 = time.perf_counter()
    print("Time taken:", round(t1 - t0, 2), "seconds")
