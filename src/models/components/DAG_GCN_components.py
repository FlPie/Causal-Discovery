import numpy as np

import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
import pytorch_lightning as pl

import copy


class DAG_GCN_Encoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H1: int, 
                 H2: int, 
                 H3: int, 
                 out_channels: int,
                 activation: torch.nn.Module=torch.nn.ReLU(),
                 batch_norm: bool=False,
                 num_features: int=11,
                 init: bool=True,
                 dropout: float=0.0):
        super().__init__()
        if init:
            self.init = 'kaiming_uniform'
        else:
            self.init = None

        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (pyg.nn.Linear(in_channels, H1, 
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(H1),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, H2,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(H2),
                copy.deepcopy(activation), # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (pyg.nn.Linear(in_channels, H1, 
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, H2,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation), # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def set_dense_adj(self, dense_adj: np.ndarray, init: bool=False):
        self.dense_adj = torch.nn.Parameter(torch.tensor(dense_adj, dtype=torch.float), 
                                            requires_grad=True)
        if init:
            torch.nn.init.kaiming_uniform_(self.dense_adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.sinh(3.0 * self.dense_adj)
        # dense_adj = torch.nn.functional.leaky_relu(dense_adj)
        Z = self.model(x, dense_adj)
        return Z, dense_adj

        
class DAG_GCN_Decoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H3: int, 
                 H2: int, 
                 H1: int, 
                 out_channels: int, 
                 activation: torch.nn.Module=torch.nn.ReLU(),
                 batch_norm: bool=False,
                 num_features: int=11,
                 init: bool=True,
                 dropout: float=0.0):
        super().__init__()
        if init:
            self.init = 'kaiming_uniform'
        else:
            self.init = None
        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),  # TODO: Need or not?
                # MLP
                (pyg.nn.Linear(H2, H1,
                                weight_initializer=self.init,
                                bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, out_channels,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                copy.deepcopy(activation),  # TODO: Need or not?
                # MLP
                (pyg.nn.Linear(H2, H1,
                                weight_initializer=self.init,
                                bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, out_channels,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, dense_adj: torch.Tensor) -> torch.Tensor:
        return self.model(x, dense_adj)


class DAG_GCN_VEncoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H1: int, 
                 H2: int, 
                 H3: int, 
                 out_channels: int,
                 activation: torch.nn.Module=torch.nn.ReLU(),
                 batch_norm: bool=False,
                 num_features: int=11,
                 dropout: float=0.0):
        super().__init__()
        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                (pyg.nn.DenseGCNConv(in_channels, H1), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H1, H2), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation), # TODO: Need or not?
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (torch.nn.Linear(in_channels, H1), 'x -> x'),
                            # weight_initializer="kaiming_uniform",
                            # bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (torch.nn.Linear(H1, H2), 'x -> x'),
                            # weight_initializer="kaiming_uniform",
                            # bias_initializer="zeros"), 'x -> x'),
                # copy.deepcopy(activation), # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])
        self.gcn_mu = pyg.nn.Sequential('x, dense_adj', [
            (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),])
        self.gcn_logstd = pyg.nn.Sequential('x, dense_adj', [
            (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def set_dense_adj(self, dense_adj: np.ndarray, init: bool=False):
        self.dense_adj = torch.nn.Parameter(torch.tensor(dense_adj, dtype=torch.float), 
                                            requires_grad=True)
        if init:
            torch.nn.init.kaiming_uniform_(self.dense_adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.sinh(3.0 * self.dense_adj)
        X = self.model(x, dense_adj)

        mu = self.gcn_mu(X, dense_adj)
        logstd = self.gcn_logstd(X, dense_adj)

        return mu, logstd, dense_adj

class DAG_GCN_VDecoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H3: int, 
                 H2: int, 
                 H1: int, 
                 out_channels: int, 
                 activation: torch.nn.Module=torch.nn.ReLU(),
                 batch_norm: bool=False,
                 num_features: int=11,
                 dropout: float=0.0):
        super().__init__()
        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),  # TODO: Need or not?
                (pyg.nn.DenseGCNConv(H2, H1), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),  # TODO: Need or not?
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
            ])
                # MLP
            self.net_mu = pyg.nn.Sequential('x, dense_adj', [
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                # copy.deepcopy(activation),  # TODO: Need or not?
                (torch.nn.Linear(H2, H1), 'x -> x'),
                                # weight_initializer="kaiming_uniform",
                                # bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (torch.nn.Linear(H1, out_channels), 'x -> x'),
                            # weight_initializer="kaiming_uniform",
                            # bias_initializer="zeros"), 'x -> x'),
            ])
            self.net_logstd = pyg.nn.Sequential('x, dense_adj', [
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                # copy.deepcopy(activation),  # TODO: Need or not?
                (torch.nn.Linear(H2, H1), 'x -> x'),
                                # weight_initializer="kaiming_uniform",
                                # bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (torch.nn.Linear(H1, out_channels), 'x -> x'),
                            # weight_initializer="kaiming_uniform",
                            # bias_initializer="zeros"), 'x -> x'),
            ])
        self.net_mu = pyg.nn.Sequential('x, dense_adj', [
            (pyg.nn.DenseGCNConv(H1, out_channels), 'x, dense_adj -> x'),])
        self.net_logstd = pyg.nn.Sequential('x, dense_adj', [
            (pyg.nn.DenseGCNConv(H1, out_channels), 'x, dense_adj -> x'),])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, mu: torch.Tensor, logstd: torch.Tensor, 
                dense_adj: torch.Tensor) -> torch.Tensor:
        Z = mu + torch.randn_like(logstd) * torch.exp(logstd)
        Z = self.model(Z, dense_adj)
        mu_x = self.net_mu(Z, dense_adj)
        logstd_x = self.net_logstd(Z, dense_adj)
        return mu_x, logstd_x


class DAG_GNN_Encoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H1: int, 
                 out_channels: int, 
                 dropout: float=0.0):
        super().__init__()
        self.out_channels = out_channels
        self.model = pyg.nn.Sequential('x', [
            # MLP
            (torch.nn.Linear(in_channels, H1), 'x -> x'),
            torch.nn.ReLU(inplace=True),
            (torch.nn.Linear(H1, out_channels), 'x -> x'),
        ])
        self.init_weights()

    def set_dense_adj(self, dense_adj: np.ndarray):
        self.dense_adj = torch.nn.Parameter(torch.tensor(dense_adj, dtype=torch.float), 
                                            requires_grad=True)

    def set_W(self, W: torch.Tensor):
        self.W = torch.nn.Parameter(W, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.sinh(3.0 * self.dense_adj)
        Z = torch.matmul(self._preprocess_adj(dense_adj),
                         self.model(x) + self.W) - self.W
        return Z, dense_adj, self.W

    def _preprocess_adj(self, dense_adj: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.eye(dense_adj.shape[0], 
                              dtype=torch.float, 
                              device=dense_adj.device)
        - dense_adj.transpose(0, 1)
        return dense_adj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)

        
class DAG_GNN_Decoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H1: int, 
                 out_channels: int, 
                 dropout: float=0.0):
        super().__init__()
        
        self.model = pyg.nn.Sequential('x', [
            # MLP
            (torch.nn.Linear(in_channels, H1), 'x -> x'),
            torch.nn.ReLU(inplace=True),
            (torch.nn.Linear(H1, out_channels), 'x -> x'),
        ])
        
        self.init_weights()

    def forward(self, x: torch.Tensor, dense_adj: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(self._preprocess_adj(dense_adj),
                         x + W) - W
        return self.model(z)
    
    def _preprocess_adj(self, dense_adj: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.inverse(
            torch.eye(dense_adj.shape[0], 
                      dtype=torch.float,
                      device = dense_adj.device)
            - dense_adj.transpose(0, 1)
        )
        return dense_adj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)

class DAG_Diff_Encoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H1: int, 
                 H2: int, 
                 H3: int, 
                 out_channels: int,
                 activation: torch.nn.Module=torch.nn.ReLU(),
                 batch_norm: bool=False,
                 num_features: int=11,
                 init: bool=True,
                 dropout: float=0.0):
        super().__init__()
        if init:
            self.init = 'kaiming_uniform'
        else:
            self.init = None

        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (pyg.nn.Linear(in_channels, H1, 
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(H1),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, H2,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(H2),
                copy.deepcopy(activation), # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # MLP
                (pyg.nn.Linear(in_channels, H1, 
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, H2,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation), # TODO: Need or not?
                # GCN
                (pyg.nn.DenseGCNConv(H2, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, out_channels), 'x, dense_adj -> x'),
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def set_dense_adj(self, dense_adj: np.ndarray, init: bool=False):
        self.dense_adj = torch.nn.Parameter(torch.tensor(dense_adj, dtype=torch.float), 
                                            requires_grad=True)
        if init:
            torch.nn.init.kaiming_uniform_(self.dense_adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense_adj = torch.sinh(3.0 * self.dense_adj)
        # dense_adj = torch.nn.functional.leaky_relu(dense_adj)
        Z = self.model(x, dense_adj)
        return Z, dense_adj

        
class DAG_Diff_Decoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 H3: int, 
                 H2: int, 
                 H1: int, 
                 out_channels: int, 
                 activation: torch.nn.Module=torch.nn.ReLU(),
                 batch_norm: bool=False,
                 num_features: int=11,
                 init: bool=True,
                 dropout: float=0.0):
        super().__init__()
        if init:
            self.init = 'kaiming_uniform'
        else:
            self.init = None
        if batch_norm:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),  # TODO: Need or not?
                # MLP
                (pyg.nn.Linear(H2, H1,
                                weight_initializer=self.init,
                                bias_initializer="zeros"), 'x -> x'),
                pyg.nn.BatchNorm(num_features),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, out_channels,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
            ])
        else:
            self.model = pyg.nn.Sequential('x, dense_adj', [
                # GCN
                (pyg.nn.DenseGCNConv(in_channels, H3), 'x, dense_adj -> x'),
                copy.deepcopy(activation),
                (pyg.nn.DenseGCNConv(H3, H2), 'x, dense_adj -> x'),
                copy.deepcopy(activation),  # TODO: Need or not?
                # MLP
                (pyg.nn.Linear(H2, H1,
                                weight_initializer=self.init,
                                bias_initializer="zeros"), 'x -> x'),
                copy.deepcopy(activation),
                (pyg.nn.Linear(H1, out_channels,
                            weight_initializer=self.init,
                            bias_initializer="zeros"), 'x -> x'),
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, dense_adj: torch.Tensor) -> torch.Tensor:
        return self.model(x, dense_adj)