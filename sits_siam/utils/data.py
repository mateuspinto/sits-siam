import numpy as np
import torch


def get_weight(x):
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


def normalize(x, mean, std):
    x = x.copy()

    all_zero_mask = np.all(x == 0, axis=1)

    x = (x - mean) / std
    x[all_zero_mask] = 0

    return x


class SitsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe,
        max_seq_len=70,
        num_features=10,
        mean=[
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
        std=[
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

    def __len__(self):
        return self.ys.shape[0]

    def __getitem__(self, idx):
        x = self.xs[idx]
        return {
            "doy": torch.from_numpy(self.doys[idx]).long(),
            "mask": torch.from_numpy(x.sum(1) == 0),
            "x": torch.from_numpy(normalize(x, self.mean, self.std)).float(),
            "weight": torch.from_numpy(get_weight(x)).float(),
        }, torch.tensor(self.ys[idx], dtype=torch.long)


if __name__ == "__main__":
    import pandas as pd

    whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")
    train_df = whole_df[whole_df["use_bert"] == 0].reset_index(drop=True)
    train_dataset = SitsDataset(train_df, max_length=45)
    print(train_dataset[0])
