import torch


def var_loss(data):  # diversity loss
    return -torch.log(torch.sum(torch.var(data, dim=0)))


def diversity_loss(x):
    # Compute the pairwise Euclidean distance between the generated samples
    dist = torch.cdist(x, x, p=2)

    # Set the diagonal elements (i.e., distance to itself) to a very large number
    mask = torch.eye(dist.shape[0], dtype=torch.bool).to(x.device)
    dist_no_diag = torch.where(mask, float("inf") * torch.ones_like(dist), dist)

    # Compute the minimum distance between each pair of samples
    min_dist, _ = torch.min(dist_no_diag, dim=1)

    # Compute the average minimum distance as the diversity score
    div_score = torch.mean(min_dist)

    return -div_score  # to be maximised


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
