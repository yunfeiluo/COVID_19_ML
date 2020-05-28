import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class mlp_linear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mlp_linear, self).__init__()
        self.sequential_liner = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.view(len(x), -1)
        return self.sequential_liner(x)

class RNN_layers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_layers, self).__init__()
        self.hidden_size = hidden_size

        self.rnn1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))
    
    def forward(self, input_seq):
        out, self.hidden_cell = self.rnn1(input_seq)
        # print('out shape after rnn1', out.shape)
        out = self.fc1(out[:, -1, :].view(len(out), -1))
        return out


class multitask_mlp(nn.Module):
    def __init__(self, regions: list, input_size, hidden_size, output_size, regionalize=False):
        super(multitask_mlp, self).__init__()
        self.regionalize = regionalize
        print('regionalize', self.regionalize)
        self.regions = regions
        self.shared_out_size = output_size
        if self.regionalize:
            self.shared_out_size = hidden_size // 2
        #self.shared_layers = mlp_linear(input_size, hidden_size, self.shared_out_size)
        self.shared_layers = RNN_layers(input_size, hidden_size, self.shared_out_size)

        # "personalized layers for each region"
        dense_layer = dict() # map: region -> corresponding layers
        if self.regionalize:
            for region in regions:
                sequential_liner = nn.Sequential(
                    nn.Linear(self.shared_out_size, self.shared_out_size // 2),
                    nn.ReLU(),
                    nn.Linear(self.shared_out_size // 2, output_size)
                )

                dense_layer[region] = sequential_liner
        
        self.region_dense_layer = nn.ModuleDict(dense_layer)

        self.final_act = nn.ReLU()
    
    def forward(self, region, input_seq):
        shared_out = self.shared_layers(input_seq)
        if not self.regionalize:
            return self.final_act(shared_out)
        return self.final_act(self.region_dense_layer[region](shared_out))
