import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
import numpy as np

class mlp_linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_linear, self).__init__()
        self.sequential_liner = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            #nn.ReLU()
        )

    def forward(self, x):
        return self.sequential_liner(x)
    