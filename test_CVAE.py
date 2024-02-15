import torch
from src.CVAE import CVAE
import pandas as pd
import os
from datetime import datetime


def generate_samples(model, num_samples, conditions, latent_dim, device="cpu"):
    # Ensure model is in evaluation mode
    model.eval()

    # Sample from a standard normal distribution
    z = torch.randn(num_samples, latent_dim).to(device)
    conditions = conditions.to(device)

    # Create condition vectors
    # This needs to be adjusted based on how your conditions are represented
    # Here, as an example, we're using one-hot vectors for categorical conditions

    # Generate samples
    with torch.no_grad():
        generated_samples = model.decoder(z, conditions)

    return generated_samples


def load_model(model_path, input_dim, condition_dim, latent_dim, device="cpu"):
    model = CVAE(input_dim, condition_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def prepare_same_conditions():
    # import conditions generated for cGAN for a comparison
    csv_path = "./synthesis_100k/synthesis_n_100000_240104_1954.csv"
    print(f"importing conditions from {csv_path}...")
    conditions = pd.read_csv(csv_path).to_numpy()[:, -2:]
    conditions = torch.tensor(conditions).float().to(device)
    return conditions


def prepare_fixed_conditions(d_max, eta_min, num_samples):
    print(f"Preparing fixed conditions with d_max={eta_min} and eta_min={d_max}...")
    arr_d_max = torch.full((num_samples, 1), d_max).float()
    arr_eta_min = torch.full((num_samples, 1), eta_min).float()
    conditions = torch.cat((arr_d_max, arr_eta_min), 1).to(device)
    return conditions


# Parameters (adjust these as needed)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./weights/CVAE/cvae_epoch_final.pth"  # Path to your trained model
num_samples = 100000
feature_size = 5  # Feature size of your 1D data
condition_dim = 2  # Dimension of your condition variable
latent_dim = 3  # Latent space dimension

save_path = "./synthesis_100k/CVAE"
os.makedirs(save_path, exist_ok=True)

# Load the trained model
cvae_model = load_model(model_path, feature_size, condition_dim, latent_dim, device)

# prepare conditions
# d_max = 2.5
# eta_min = 1.0
# conditions = prepare_fixed_conditions(d_max, eta_min, num_samples)
conditions = prepare_same_conditions()
print("conditions size:", conditions.shape)
# Generate samples
generated_data = generate_samples(
    cvae_model, num_samples, conditions, latent_dim, device
)
generated_data = torch.cat((generated_data, conditions), 1).detach().cpu().numpy()
print("generated_data shape:", generated_data.shape)

cols = ["l2", "l3", "l4", "EE_x", "EE_y", "d_max", "eta_min"]
df = pd.DataFrame(generated_data, columns=cols)

# Get the current date and time
current_datetime = datetime.now()
# Format the date and time as per the specified format: 240124_1939
formatted_datetime = current_datetime.strftime("%y%m%d_%H%M")
# filename = f"synthesis_d_{d_max}_eta_{eta_min}_n_{num_samples}.csv"
filename = f"synthesis_cvae_{formatted_datetime}.csv"
csv_path = os.path.join(save_path, filename)
df.to_csv(csv_path, index=False)
print("Generated data is saved to", csv_path)
