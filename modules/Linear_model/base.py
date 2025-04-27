import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc1(x)