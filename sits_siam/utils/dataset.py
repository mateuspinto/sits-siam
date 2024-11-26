from typing import Dict, Tuple
from tqdm.std import tqdm

import pathlib
import numpy as np
import torch
import pandas as pd


class SitsDatasetFromDataframe(torch.utils.data.Dataset):
    """
    Dataset class for satellite image time series data from pd.DataFrame.

    Processes input DataFrame into tensors suitable for training machine learning models, including data normalization
    and computation of sample weights.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing satellite images and labels. Expected columns include 'id', 'time', 'doy', and spectral bands.
    max_seq_len : int, optional
        Maximum sequence length for time series data (default is 70).
    num_features : int, optional
        Number of features per sample (default is 10).
    mean : List[float], optional
        Mean values for normalization of each feature (default values provided).
    std : List[float], optional
        Standard deviation values for normalization of each feature (default values provided).

    Attributes
    ----------
    xs : np.ndarray
        Array of input features of shape (num_samples, max_seq_len, num_features).
    doys : np.ndarray
        Array of day-of-year values corresponding to each time step.
    ys : np.ndarray
        Array of labels corresponding to each sample.

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_seq_len: int = 70,
        num_features: int = 10,
        transform=None,
    ):
        BANDS = [
            "blue",
            "green",
            "red",
            "red_edge_1",
            "red_edge_2",
            "red_edge_3",
            "nir",
            "red_edge_4",
            "swir_1",
            "swir_2",
        ]

        dataframe = dataframe.sort_values(["id", "time"])

        ids = dataframe["id"].to_numpy()
        times = dataframe["time"].astype(int).to_numpy()
        doys = dataframe["doy"].to_numpy()
        bands_data = dataframe[BANDS].to_numpy()

        unique_ids, id_indices = np.unique(ids, return_inverse=True)
        num_ids = len(unique_ids)

        self.xs = np.zeros((num_ids, max_seq_len, num_features), dtype=np.half)
        self.doys = np.zeros((num_ids, max_seq_len), dtype=np.int16)

        self.xs[id_indices, times, :] = bands_data
        self.doys[id_indices, times] = doys

        labels_df = dataframe[["id", "label"]].drop_duplicates("id").set_index("id")
        self.ys = labels_df.loc[unique_ids, "label"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.ys.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get the sample corresponding to the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            A tuple containing:
            - A dictionary with keys:
                'doy': torch.Tensor of day-of-year values,
                'mask': torch.Tensor indicating missing data,
                'x': torch.Tensor of normalized input features,
                'weight': torch.Tensor of sample weights.
            - The corresponding label tensor.
        """
        sample = {"doy": self.doys[idx], "x": self.xs[idx], "y": self.ys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SitsDatasetFromFormerFormat(object):
    def __init__(self, folder_path, max_seq_len, transform=None, limit=None):
        filenames = sorted(pathlib.Path(folder_path).glob("*.npz"))

        size = len(filenames) if limit is None else limit
        self.xs = np.zeros((size * 25, max_seq_len, 10), dtype=np.half)
        self.doys = np.zeros((size * 25, max_seq_len), dtype=np.int16)
        self.transform = transform

        for id in tqdm(range(size)):
            filename = filenames[id]
            data = np.load(filename)
            ts = data["ts"]  # Shape: (seq_len, 10, 5, 5)
            doy = data["doy"]  # Shape: (seq_len,)

            seq_len = ts.shape[0]
            if seq_len > max_seq_len:
                seq_len = max_seq_len

            ts_reshaped = ts[:seq_len].transpose(2, 3, 0, 1).reshape(-1, seq_len, 10)
            doy_replicated = np.tile(doy[:seq_len], (25, 1))

            start_idx = id * 25
            end_idx = start_idx + 25

            self.xs[start_idx:end_idx, :seq_len, :] = ts_reshaped
            self.doys[start_idx:end_idx, :seq_len] = doy_replicated

    def __len__(self) -> int:
        return self.ys.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sample = {"doy": self.doys[idx], "x": self.xs[idx], "y": 0}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    import pandas as pd

    whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")
    train_df = whole_df[whole_df["use_bert"] == 0].reset_index(drop=True)
    train_dataset = SitsDatasetFromDataframe(train_df, max_seq_len=45)
    print(train_dataset[0])
