import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_convergence(
    G_losses, D_real_losses, D_fake_losses, real_accuracies, fake_accuracies, lr_G, lr_D, weights_path
):
    lst_epoch = np.array(range(len(G_losses)))
    plt.figure(figsize=(10, 5))
    plt.title(f"G and D Loss lrg={lr_G} lrd={lr_D}")
    plt.plot(lst_epoch, G_losses, label="G")
    plt.plot(lst_epoch, D_real_losses, label="D_real")
    plt.plot(lst_epoch, D_fake_losses, label="D_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    filename = os.path.join(weights_path, "loss_G_D_plot.png")
    plt.savefig(filename, dpi=200)
    # print('Loss plot saved to', filename)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title(f"D Acc lrg={lr_G} lrd={lr_D}")
    plt.plot(lst_epoch, real_accuracies, label="acc_real")
    plt.plot(lst_epoch, fake_accuracies, label="acc_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    filename = os.path.join(weights_path, "acc_plot.png")
    plt.savefig(filename, dpi=200)
    # print('Accuracy plot saved to', filename)
    plt.close()

    return


def plot_loss_P_Var(P_losses, Div_losses, w_P, w_div, weights_path):
    roll = 5
    df = pd.DataFrame({"p": P_losses, "var": Div_losses}).rolling(roll).mean()

    lst_epoch = np.array(range(len(P_losses)))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_title(f"P and Var Loss w_P={w_P} w_div={w_div} rolling mean of {roll}")
    ax1 = ax.twinx()
    ax.plot(lst_epoch, df["p"], "g-")
    ax1.plot(lst_epoch, df["var"], "b-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("P_loss", color="g")
    ax1.set_ylabel("Var_loss", color="b")
    # ax.set_ylim([0, 500])
    # ax1.set_ylim([0, 5])
    filename = os.path.join(weights_path, "loss_P_Var_plot.png")
    plt.savefig(filename)
    plt.close()