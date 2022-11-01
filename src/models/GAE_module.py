import pandas as pd
import numpy as np

from tqdm import tqdm

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.cm as cm

import torch
import torch_geometric as pyg
import pytorch_lightning as pl

import wandb

from .components.GAE_components import MLP
from .components import DAG_utils as DAG_utils


class GAE(pl.LightningModule):
    def __init__(self,
                 input_dim: int=1,
                 num_hidden_layer: int=3,
                 hidden_dim: int=16,
                 output_dim: int=1,
                 activation=torch.nn.ReLU(),
                 optimizer: torch.optim.Optimizer=None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler=None,
                 graph_threshold: float=0.3,
                 l1_penalty: float=0.0,
                 alpha: float=0.0,
                 beta: float=2.0,
                 gamma: float=0.25,
                 rho: float=1.0,
                 rho_threshold: float=1e30,
                 h_threshold: float=1e-8,
                 init_iter: int=3,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = MLP(input_dim=input_dim,
                           num_layers=num_hidden_layer,
                           hidden_dim=hidden_dim,
                           output_dim=hidden_dim,
                           activation=activation)
        self.gcn = pyg.nn.Sequential('x, dense_adj', [
            # GCN
            (pyg.nn.DenseGCNConv(hidden_dim, hidden_dim), 'x, dense_adj -> x'),
            torch.nn.ReLU(inplace=True),
            (pyg.nn.DenseGCNConv(hidden_dim, hidden_dim), 'x, dense_adj -> x'),
        ])
        self.decoder = MLP(input_dim=hidden_dim,
                           num_layers=num_hidden_layer,
                           hidden_dim=hidden_dim,
                           output_dim=output_dim,
                           activation=activation)
        np_adj = np.random.uniform(low=-0.5, high=0.5,
                                   size=(self.hparams.num_features, 
                                         self.hparams.num_features))
        self.dense_adj = torch.nn.Parameter(torch.tensor(np_adj, dtype=torch.float), 
                                            requires_grad=True)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Read adj_cols from raw data
        self.adj_cols = pd.read_csv(self.hparams.raw_path, nrows=0).columns
        # Read gt from gt file and set labels as raw data
        self.gt_df = pd.read_csv(self.hparams.gt_path, index_col=0)
        self.gt_df = self.gt_df[self.adj_cols]
        self.gt_df = self.gt_df.reindex(self.adj_cols)
        self.ground_truth_G = nx.from_pandas_adjacency(self.gt_df, create_using=nx.DiGraph)

        # hparams
        self.graph_threshold = graph_threshold
        self.l1_penalty = l1_penalty
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.rho_threshold = rho_threshold
        self.h_threshold = h_threshold

        # update params
        self.h = np.inf
        self.h_new = np.inf

    def forward(self, x):
        H = self.encoder(x)
        H_prime = self.gcn(H, self.dense_adj)
        X_hat = self.decoder(H_prime)

        MSE_loss = torch.nn.MSELoss()(x.squeeze(), X_hat.squeeze())

        return MSE_loss

    def step(self, batch, batch_idx):
        MSE_loss = self.forward(batch[batch_idx])
        loss, current_h = self._loss(MSE_loss, self.dense_adj)
        return loss, MSE_loss, current_h

    def on_train_start(self):
        gt_graph_img, gt_adj_img = DAG_utils.get_plot_imgs(self.gt_df, self.ground_truth_G, self.ground_truth_G)
        wandb.log({'gt/graph': wandb.Image(gt_graph_img)})
        wandb.log({'gt/adj': wandb.Image(gt_adj_img)})

    def training_step(self, batch, batch_idx):
        loss, self.MSE_loss, self.h_new = self.step(batch, batch_idx)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/MSE_loss', self.MSE_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/h', self.h_new, on_step=False, on_epoch=True, prog_bar=False)
        return {'loss': loss,}

    def training_epoch_end(self, outputs) -> None:
        graph = self.dense_adj.cpu().clone().detach().numpy()
        graph[np.abs(graph) < self.graph_threshold] = 0
        G = nx.DiGraph(graph)
        mapping = dict(zip(G, self.adj_cols))
        nx.relabel_nodes(G, mapping, copy=False)

        accu = DAG_utils.count_accuracy(self.ground_truth_G, G)
        self.log('train/accu', accu, on_step=False, on_epoch=True, prog_bar=False)
        self.log('shd', accu['shd'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('tpr', accu['tpr'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('fdr', accu['fdr'], on_step=False, on_epoch=True, prog_bar=False)

        if self.current_epoch % self.hparams.plot_every == 0:
            graph_img, adj_img = DAG_utils.get_plot_imgs(graph, G, self.ground_truth_G)
            wandb.log({"train/graph": wandb.Image(graph_img)})
            wandb.log({"train/adj": wandb.Image(adj_img)})

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        return optimizer
    
    def _loss(self, MSE_loss, dense_adj):
        current_h = self._compute_h(dense_adj)
        loss = ((0.5 / dense_adj.shape[0]) * MSE_loss
                + self.l1_penalty * torch.norm(dense_adj, p=1)
                + self.alpha * current_h
                + 0.5 * self.rho * current_h * current_h)
        return loss, current_h
    
    def _compute_h(self, dense_adj):
        h = torch.trace(torch.matrix_exp(dense_adj * dense_adj)) - dense_adj.shape[0]
        return h