import torch
import numpy as np
import torch.nn.functional as F

def gen_random_A(batch_size, measurement_size, num_nodes):
    return torch.from_numpy(
                np.random.normal(loc=0, scale=1/measurement_size, size=[batch_size, measurement_size, num_nodes])
            ).float()


def gen_binary_A(batch_size, measurement_size, num_nodes):
    A = torch.zeros([batch_size, measurement_size, num_nodes])
    for i in range(batch_size):
        B = np.random.choice(num_nodes, size=[measurement_size], replace=False)
        for j in range(measurement_size):
            A[i, j, B[j]] = 1
    return A


def optimize_latent_rep(model, A, measurements, latent_channels, device):
    # set up optimizer w.r.t. latent representation
    latent_rep = torch.zeros([measurements.shape[0], latent_channels])
    latent_rep.requires_grad_(True)
    latent_rep.to(device)
    optimizer = torch.optim.Adam([latent_rep],
                                 lr=0.1)
    #optimize
    for i in range(1000):
        optimizer.zero_grad()
        out = model.decode(latent_rep)
        loss = F.l1_loss(torch.matmul(A, out), measurements, reduction='mean')
        loss.backward()
        optimizer.step()
    return out


def eval_reconstruction(pred, true, std, mean):
    # Denormalize
    pred = (pred * std) + mean
    true = (true * std) + mean

    # scale to mm
    pred *= 1000
    true *= 1000

    tmp_error = torch.sqrt(
        torch.sum((pred - true) ** 2,
                  dim=2))  # [num_graphs, num_nodes]

    mean_error = tmp_error.view((-1,)).mean()
    std_error = tmp_error.view((-1,)).std()
    median_error = tmp_error.view((-1,)).median()

    return mean_error, std_error, median_error