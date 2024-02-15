import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def generate_synthetic_data(data, N, k):
    """
    data: Original dataset (n_samples, n_variables)
    N: Number of synthetic samples to generate
    k: Number of nearest neighbors
    """
    n_samples, n_variables = data.shape

    # Initialize the synthetic data array
    synthetic = np.zeros((N, n_variables))

    # Initialize NearestNeighbors and fit the data
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)

    # Randomly select sample indices
    random_sample_indices = np.random.randint(low=0, high=n_samples, size=N)

    # Find k-nearest neighbors for the selected indices
    _, nn_indices = neigh.kneighbors(data[random_sample_indices])

    # Randomly choose one neighbor for each sample
    chosen_neighbor_indices = nn_indices[np.arange(N), np.random.randint(k, size=N)]

    # Generate gaps
    gaps = np.random.rand(N, 1)

    # Vectorized operation to calculate synthetic samples
    synthetic = data[random_sample_indices] + gaps * (
        data[chosen_neighbor_indices] - data[random_sample_indices]
    )

    arr_d = torch.from_numpy(synthetic[:, 0].reshape(-1, 1)).float()
    arr_gr_min = torch.from_numpy(synthetic[:, 1].reshape(-1, 1)).float()
    return arr_d, arr_gr_min
