from typing import List, Dict, Tuple

import numpy as np
import torch
import pandas as pd


def get_weight_for_ndvi_word2vec(x: np.ndarray) -> np.ndarray:
    """
    Compute sample weights based on NDVI from input spectral data.

    This function calculates the Normalized Difference Vegetation Index (NDVI) from the input spectral data,
    applies cloud and dark pixel masking, and computes weights based on the NDVI values. The weights are then
    normalized to sum to 1.

    Parameters
    ----------
    x : np.ndarray
        Input spectral data array of shape (n_samples, n_features). The array should include at least the following bands:
        - x[:, 0]: Blue
        - x[:, 1]: Green
        - x[:, 2]: Red
        - x[:, 6]: Near-Infrared (NIR)
        - x[:, 8]: Shortwave Infrared 1 (SWIR1)
        - x[:, 9]: Shortwave Infrared 2 (SWIR2)

    Returns
    -------
    np.ndarray
        Normalized weights array of shape (n_samples,).

    """
    all_zero_mask = np.all(x == 0, axis=1)

    score = np.ones(x.shape[0])
    score = np.minimum(score, (x[:, [0, 1, 2]].sum(1) - 0.2) / 0.6)  # rgb
    cloud = score * 100 > 20
    dark = x[:, [6, 8, 9]].sum(1) < 0.35  # NIR, SWIR1, SWIR2

    ndvi = (x[:, 6] - x[:, 2]) / (x[:, 6] + x[:, 2] + 1e-6)
    ndvi[cloud] = -1
    ndvi[dark] = -1
    ndvi = ndvi.clip(-1, 1)

    weight = np.exp(ndvi)
    weight /= weight.sum()

    weight[all_zero_mask] = 0

    return weight


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize input data using provided mean and standard deviation.

    Rows where all elements are zero are set to zero after normalization to prevent division by zero or NaN values.

    Parameters
    ----------
    x : np.ndarray
        Input data array to be normalized.
    mean : np.ndarray
        Mean values for each feature used in normalization.
    std : np.ndarray
        Standard deviation values for each feature used in normalization.

    Returns
    -------
    np.ndarray
        Normalized data array.

    """
    x = x.copy()

    all_zero_mask = np.all(x == 0, axis=1)

    x = (x - mean) / std
    x[all_zero_mask] = 0

    return x


class SitsDataset(torch.utils.data.Dataset):
    """
    Dataset class for satellite image time series data.

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
        mean: List[float] = [
            0.0656,
            0.0948,
            0.1094,
            0.1507,
            0.2372,
            0.2673,
            0.2866,
            0.2946,
            0.2679,
            0.1985,
        ],
        std: List[float] = [
            0.036289,
            0.043310,
            0.064736,
            0.057953,
            0.074167,
            0.096407,
            0.097816,
            0.098368,
            0.089847,
            0.097866,
        ],
    ):
        self.mean = np.array(
            [mean],
            dtype=np.half,
        )
        self.std = np.array(
            std,
            dtype=np.half,
        )

        bands = [
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
        bands_data = dataframe[bands].to_numpy()

        unique_ids, id_indices = np.unique(ids, return_inverse=True)
        num_ids = len(unique_ids)

        self.xs = np.zeros((num_ids, max_seq_len, num_features), dtype=np.half)
        self.doys = np.zeros((num_ids, max_seq_len), dtype=np.int16)

        self.xs[id_indices, times, :] = bands_data
        self.doys[id_indices, times] = doys

        labels_df = dataframe[["id", "label"]].drop_duplicates("id").set_index("id")
        self.ys = labels_df.loc[unique_ids, "label"].to_numpy()

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
        x = self.xs[idx]
        return {
            "doy": torch.from_numpy(self.doys[idx]).long(),
            "mask": torch.from_numpy(x.sum(1) == 0),
            "x": torch.from_numpy(normalize(x, self.mean, self.std)).float(),
            "weight": torch.from_numpy(get_weight_for_ndvi_word2vec(x)).float(),
        }, torch.tensor(self.ys[idx], dtype=torch.long)


if __name__ == "__main__":
    import pandas as pd

    whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")
    train_df = whole_df[whole_df["use_bert"] == 0].reset_index(drop=True)
    train_dataset = SitsDataset(train_df, max_seq_len=45)
    print(train_dataset[0])
