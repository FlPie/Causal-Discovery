import numpy as np

import torch
import torch_geometric as pyg
import pytorch_lightning as pl

import copy


class MLP(torch.nn.Module):
    def __init__(self, 
                 input_dim: int=1, 
                 num_layers: int=3, 
                 hidden_dim: int=16, 
                 output_dim: int=1,
                 activation=torch.nn.ReLU()):
        super(MLP, self).__init__()
        # self.desc = desc
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation

        mlp = []
        for i in range(self.num_layers):
            input_size = hidden_dim
            if i == 0:
                input_size = input_dim
            weight = torch.nn.Linear(in_features=input_size,
                               out_features=self.hidden_dim)
            mlp.append(weight)
            if activation is not None:
                mlp.append(copy.deepcopy(activation))
        out_layer = torch.nn.Linear(in_features=self.hidden_dim,
                                    out_features=self.output_dim)
        mlp.append(out_layer)

        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)