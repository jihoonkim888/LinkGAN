import pandas as pd
from torch.utils.data import DataLoader

def import_data(data_path):
    df = (
        pd.read_csv(data_path).to_numpy().astype(float)
    )  # read csv file with no header col and l1
    return df


def get_dataloader(input_tensors, batch_size, workers):
    dataloader = DataLoader(
        input_tensors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    return dataloader