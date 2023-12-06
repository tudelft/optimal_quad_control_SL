import torch
from torch import nn

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def forward(self, x):
        return (x - self.mean)/self.std
    
class MapToRange(nn.Module):
    def __init__(self, min_, max_):
        super().__init__()
        self.min = torch.tensor(min_, dtype=torch.float32)
        self.max = torch.tensor(max_, dtype=torch.float32)

    def forward(self, x):
        return self.min + x*(self.max - self.min)