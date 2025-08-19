from typing import Dict, Tuple
from tqdm.std import tqdm

import pathlib
import numpy as np
import geopandas as gpd
import pandas as pd
import torch
import pandas as pd
from typing import Union
import os


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


class SitsDatasetFromFormerFormat(torch.utils.data.Dataset):
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


class SitsPretrainDatasetFromNpz(torch.utils.data.Dataset):
    def __init__(
        self, npz_dir: Union[pathlib.Path, str], world_size: int = 1, transform=None
    ):
        if isinstance(npz_dir, str):
            npz_dir = pathlib.Path(npz_dir)

        self.transform = transform
        # Detect the local rank from the environment variable
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.local_rank = local_rank
        self.world_size = world_size

        # List all .npz files and ensure divisibility by world_size
        self.npz_files = sorted(npz_dir.glob("*.npz"))
        total_files = len(self.npz_files)

        # Adjust the number of files to be divisible by world_size
        if total_files % world_size != 0:
            total_files = (total_files // world_size) * world_size
            self.npz_files = self.npz_files[:total_files]

        files_per_gpu = total_files // world_size

        # Assign a subset of files to each GPU
        start_file_idx = local_rank * files_per_gpu
        end_file_idx = start_file_idx + files_per_gpu
        self.npz_files = self.npz_files[start_file_idx:end_file_idx]

        # Preallocate arrays for 'ts' and 'doys'
        num_samples_per_file = 100000 * 25  # Based on your assumption
        total_samples = num_samples_per_file * len(self.npz_files)
        self.ts = np.zeros((total_samples, 45, 10), dtype=np.float16)
        self.doys = np.zeros((total_samples, 45), dtype=np.int16)

        # Load data into RAM for this GPU
        for n, npz_file_path in tqdm(
            enumerate(self.npz_files),
            total=len(self.npz_files),
            desc=f"Loading dataset into RAM on GPU {local_rank}...",
        ):
            data = np.load(npz_file_path)
            start_idx = n * num_samples_per_file
            end_idx = start_idx + num_samples_per_file
            self.ts[start_idx:end_idx] = data["ts"].astype(np.float16)
            self.doys[start_idx:end_idx] = data["doys"].astype(np.int16)

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx: int):
        sample = {"x": self.ts[idx], "doy": self.doys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __add__(self, other):
        if not isinstance(other, SitsPretrainDatasetFromNpz):
            raise ValueError("Can only add SitsPretrainDatasetFromNpz objects together")

        combined_dataset = SitsPretrainDatasetFromNpz(
            npz_dir=pathlib.Path("."), world_size=self.world_size
        )
        combined_dataset.ts = np.concatenate((self.ts, other.ts), axis=0)
        combined_dataset.doys = np.concatenate((self.doys, other.doys), axis=0)
        return combined_dataset


class SitsFinetuneDatasetFromNpz(torch.utils.data.Dataset):
    def __init__(self, npz_file: Union[pathlib.Path, str], transform=None):
        if isinstance(npz_file, str):
            npz_file = pathlib.Path(npz_file)

        data = np.load(npz_file)
        self.ts = data["ts"].astype(np.float16)
        self.doys = data["doys"].astype(np.int16)
        self.ys = data["ys"].astype(np.int16)
        self.transform = transform

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx: int):
        sample = {"x": self.ts[idx], "doy": self.doys[idx], "y": self.ys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @property
    def num_classes(self):
        return np.max(self.ys) + 1

    @property
    def sequence_length(self):
        return self.ts.shape[1]


class AgriGEELiteDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gdf: str | gpd.GeoDataFrame,
        sits_df: str | pd.DataFrame,
        band_order=None,
        timestamp_processing="doy",
        transform=None,
    ):
        if isinstance(gdf, str):
            self.gdf = gpd.read_parquet(gdf)
        else:
            self.gdf = gdf.copy()

        if isinstance(sits_df, str):
            self.sits_df = pd.read_parquet(sits_df)
        else:
            self.sits_df = sits_df.copy()

        self.gdf = self.gdf.reset_index().rename(columns={"index": "indexnum"})
        self.transform = transform

        self.band_order = (
            band_order
            if band_order
            else [
                "blue",
                "green",
                "red",
                "re1",
                "re2",
                "re3",
                "nir",
                "re4",
                "swir1",
                "swir2",
            ]
        )

        self.timestamp_processing = timestamp_processing
        self.max_seq_len = self.sits_df.groupby("indexnum").size().max()

        self.sits_df["timestamp"] = pd.to_datetime(self.sits_df["timestamp"])
        self.sequence_length = self.max_seq_len

        if self.timestamp_processing == "doy":
            self.sits_df["timestamp"] = self.sits_df["timestamp"].dt.dayofyear
        elif self.timestamp_processing == "days_after_start":
            start_map = self.gdf.set_index("indexnum")["start_date"]
            starts = self.sits_df["indexnum"].map(start_map)
            self.sits_df["timestamp"] = (self.sits_df["timestamp"] - starts).dt.days
            self.sits_df["timestamp"] = self.sits_df["timestamp"] + 1

        self.xs, self.doys = self.to_numpy_arrays_wo_for()
        self.ys = self.gdf["crop_number"].values
        self.num_classes = int(self.gdf["crop_class"].nunique())

    def to_numpy_arrays_wo_for(self):
        n_samples = len(self.gdf)
        n_bands = len(self.band_order)

        X = np.full((n_samples, self.max_seq_len, n_bands), 0, dtype=np.float16)
        T = np.full((n_samples, self.max_seq_len), 0, dtype=np.int16)

        sits_sorted = self.sits_df.sort_values(["indexnum", "timestamp"]).copy()

        index_map = {idx: i for i, idx in enumerate(self.gdf["indexnum"].values)}
        sits_sorted["parcel_idx"] = sits_sorted["indexnum"].map(index_map)

        sits_sorted["seq_idx"] = sits_sorted.groupby("indexnum").cumcount()

        sits_sorted = sits_sorted[sits_sorted["seq_idx"] < self.max_seq_len]

        band_values = sits_sorted[self.band_order].to_numpy()

        time_values = sits_sorted["timestamp"].to_numpy()

        pi = sits_sorted["parcel_idx"].to_numpy()
        si = sits_sorted["seq_idx"].to_numpy()

        X[pi, si, :] = band_values
        T[pi, si] = time_values

        return X, T

    def __len__(self) -> int:
        return self.ys.shape[0]

    def __getitem__(self, idx: int):
        sample = {"doy": self.doys[idx], "x": self.xs[idx], "y": self.ys[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    import pandas as pd

    whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")
    train_df = whole_df[whole_df["use_bert"] == 0].reset_index(drop=True)
    train_dataset = SitsDatasetFromDataframe(train_df, max_seq_len=45)
    print(train_dataset[0])
