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

from .components.DAG_GCN_components import DAG_GCN_Encoder, DAG_GCN_Decoder
from .components import DAG_utils as DAG_utils


class DAG_Diffution(pl.LightningModule):
    def __init__(self,
                 encoder: DAG_GCN_Encoder,
                 decoder: DAG_GCN_Decoder,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler=None,
                 lambda_A: float=0.0,
                 c_A: int=1,
                 eta: int=10,
                 gamma: float=0.25,
                 graph_threshold: float=0.3,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dense_adj = np.random.uniform(low=self.hparams.adj_low, 
                                           high=self.hparams.adj_high,
                                           size=(self.hparams.num_features,self.hparams.num_features))
        self.encoder = encoder
        self.encoder.set_dense_adj(self.dense_adj, init=self.hparams.init)
        self.decoder = decoder
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
        self.h_A = np.inf
        self.h_A_new = torch.tensor(1.)
        self.h_loss = np.inf

        self.lambda_A = lambda_A
        self.c_A = c_A

        self.eta = eta
        self.gamma = gamma
        self.graph_threshold = graph_threshold

        self.best_shd_G = self.ground_truth_G
        self.best_shd = np.inf
        self.best_tpr_G = self.ground_truth_G
        self.best_tpr = 0
        self.best_loss_G = self.ground_truth_G
        self.best_loss = np.inf
        self.last_G = self.ground_truth_G

    def forward(self, X):
        Z, self.dense_adj = self.encoder(X)
        X_hat = self.decoder(Z, self.dense_adj)
        return Z, self.dense_adj, X_hat
    
    def step(self, batch, batch_idx):
        Z, self.dense_adj, X_hat = self.forward(batch[batch_idx])
        loss, losses = self.loss(batch[batch_idx].squeeze(), 
                                 Z.squeeze(), 
                                 self.dense_adj, 
                                 X_hat.squeeze())
        return Z, self.dense_adj, X_hat, loss, losses

    def on_train_start(self):
        gt_graph_img, gt_adj_img = DAG_utils.get_plot_imgs(self.gt_df, self.ground_truth_G, self.ground_truth_G)
        wandb.log({'gt/graph': wandb.Image(gt_graph_img)})
        wandb.log({'gt/adj': wandb.Image(gt_adj_img)})
        # init bests
        self.best_shd = np.inf
        self.best_tpr = 0
        self.best_loss = np.inf

    def training_step(self, batch, batch_idx):
        Z, self.dense_adj, X_hat, loss, losses = self.step(batch, batch_idx)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/losses', losses, on_step=False, on_epoch=True, prog_bar=False)
        # update best
        return {"loss": loss}

    def training_epoch_end(self, outputs):
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
            
        # update best
        loss  = np.sum(d['loss'] for d in outputs) / (len(outputs)) # * self.hparams.batch_size)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_G = G
        if accu['shd'] < self.best_shd:
            self.best_shd_G = G
            self.best_shd = accu['shd']
        if accu['tpr'] > self.best_tpr:
            self.best_tpr_G = G
            self.best_tpr = accu['tpr']
        self.last_G = G

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def loss(self, X, Z, dense_adj, X_hat):
        # reconstruction accuracy loss
        NLL_loss = self._nll_gaussian(X, X_hat)
        # KL loss
        KL_loss = self._kl_gaussian_sem(Z)
        # ELBO loss
        loss = KL_loss + NLL_loss
        # sparse loss
        sparse_loss = self._sparse_loss(dense_adj)
        # compute h(A)
        h_A = self._h_A(dense_adj)
        loss += (
            self.lambda_A * h_A
            + 0.5 * self.c_A * h_A * h_A
            + 100 * torch.trace(dense_adj * dense_adj)
            + sparse_loss
        )
        losses = {
            "ELBO_loss": NLL_loss + KL_loss,
            "NLL_loss": NLL_loss,
            "KL_loss": KL_loss,
            "MSE_loss": torch.nn.MSELoss()(X, X_hat),
        }
        return loss, losses

    def _nll_gaussian(self, X, X_hat, variance=0.0, add_const=False):
        # DAG-GNN paper equation (9)
        neg_log_p = variance + torch.div(torch.pow(X_hat - X, 2), 2.0 * np.exp(2.0 * variance))
        if add_const:
            const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
            neg_log_p += const
        return torch.mean(neg_log_p)
    
    def _kl_gaussian_sem(self, Z):
        # DAG-GNN paper equation (8)
        return 0.5 * (torch.sum(Z**2, dim=0) / Z.size(0))

    def _sparse_loss(self, adj, tau=0.1):
        return tau * torch.sum(torch.abs(adj))
    
    def _h_A(self, adj):
        # DAG-GNN paper (13)
        m = adj.shape[0]
        x = torch.eye(m).float().type_as(adj) + torch.div(adj*adj, m)
        matrix_poly = torch.matrix_power(x, m)
        expm_A = matrix_poly
        h_A = torch.trace(expm_A) - m
        return h_A
