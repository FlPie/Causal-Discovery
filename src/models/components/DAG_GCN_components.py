import numpy as np

import torch
import torch_geometric as pyg
import pytorch_lightning as pl


class DAG_GCN_Encoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H1: int, 
                 H2: int, 
                 H3: int, 
                 out_channels: int, 
                 dropout: float=0.0):
        super().__init__()
        self.model = pyg.nn.Sequential('x, dense_adj', [
            # MLP
            (torch.nn.Linear(in_channels, H1), 'x -> x'),
            torch.nn.ReLU(inplace=True),
            (torch.nn.Linear(H1, H2), 'x -> x'),
            # torch.nn.ReLU(inplace=True), # TODO: Need or not?
            # GCN
            (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
            # torch.nn.ReLU(inplace=True),
            (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
        ])

    def set_dense_adj(self, dense_adj: np.ndarray):
        self.dense_adj = torch.nn.Parameter(torch.tensor(dense_adj, dtype=torch.float), 
                                            requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.sinh(3.0 * self.dense_adj)
        Z = self.model(x, dense_adj)
        return Z, dense_adj

        
class DAG_GCN_Decoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H3: int, 
                 H2: int, 
                 H1: int, 
                 out_channels: int, 
                 dropout: float=0.0):
        super().__init__()
        
        self.model = pyg.nn.Sequential('x, dense_adj', [
            # GCN
            (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
            # torch.nn.ReLU(inplace=True),
            (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
            # torch.nn.ReLU(inplace=True), # TODO: Need or not?
            # MLP
            (torch.nn.Linear(H2, H1), 'x -> x'),
            torch.nn.ReLU(inplace=True),
            (torch.nn.Linear(H1, out_channels), 'x -> x'),
        ])

    def forward(self, x: torch.Tensor, dense_adj: torch.Tensor) -> torch.Tensor:
        return self.model(x, dense_adj)