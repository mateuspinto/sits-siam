import torch.nn as nn
import torch


class NDFIWord2VecBottleneck(nn.Module):
    def __init__(self):
        super(NDFIWord2VecBottleneck, self).__init__()

    def forward(self, features: torch.Tensor, weight: torch.Tensor):
        weight /= weight.sum(1, keepdim=True)
        features = torch.bmm(weight.unsqueeze(1), features).squeeze()

        return features
