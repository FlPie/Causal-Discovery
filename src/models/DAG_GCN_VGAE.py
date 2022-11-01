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

from .components.DAG_GCN_components import DAG_GCN_VEncoder, DAG_GCN_VDecoder
from .components import DAG_utils as DAG_utils


class DAG_GCN_VGAE(pl.LightningModule):
    def __init__(self,
                 encoder: DAG_GCN_VEncoder,
                 decoder: DAG_GCN_VDecoder,
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

        self.save_hyperparameters(ignore=['encoder', 'decder'])

    def forward(self, X):
        mu, logstd, self.dense_adj = self.encoder(X)
        mu_x, logstd_x = self.decoder(mu, logstd, self.dense_adj)
        return mu, logstd, self.dense_adj, mu_x, logstd_x
    
    def step(self, batch, batch_idx):
        mu, logstd, self.dense_adj, mu_x, logstd_x = self.forward(batch[batch_idx])
        loss, losses = self.loss(batch[batch_idx].squeeze(), 
                                 mu.squeeze(), 
                                 logstd.squeeze(),
                                 mu_x.squeeze(),
                                 logstd_x.squeeze(),
                                 self.dense_adj,
                                 )
        return mu, logstd, self.dense_adj, mu_x, logstd_x, loss, losses

    def on_train_start(self):
        gt_graph_img, gt_adj_img = DAG_utils.get_plot_imgs(self.gt_df, self.ground_truth_G, self.ground_truth_G)
        wandb.log({'gt/graph': wandb.Image(gt_graph_img)})
        wandb.log({'gt/adj': wandb.Image(gt_adj_img)})

    def training_step(self, batch, batch_idx):
        mu, logstd, self.dense_adj, mu_x, logstd_x, loss, losses = self.step(batch, batch_idx)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/losses', losses, on_step=False, on_epoch=True, prog_bar=False)
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

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def loss(self, X, mu, logstd, mu_x, logstd_x, dense_adj):
        # reconstruction accuracy loss
        NLL_loss = self._nll_gaussian(X, mu_x, logstd_x)
        # Binary_CE = self._binary_cross_entropy(X, mu_x, logstd_x)
        # KL loss
        KL_loss = self._kl_gaussian(mu, logstd)
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
        X_hat = mu_x + torch.randn_like(logstd_x) * torch.exp(logstd_x)
        losses = {
            "ELBO_loss": NLL_loss + KL_loss,
            "NLL_loss": NLL_loss,
            "KL_loss": KL_loss,
            "MSE_loss": torch.nn.MSELoss()(X, X_hat),
        }
        # loss = Binary_CE + KL_loss
        return loss, losses

    def _nll_gaussian(self, X, mu_x, logstd_x, add_const=False):
        # DAG-GNN paper equation (9)
        neg_log_p = logstd_x + torch.div(torch.pow(X - mu_x, 2), 
                                         2.0 * torch.exp(2.0 * logstd_x))
        if add_const:
            const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * logstd_x)
            neg_log_p += const
        return torch.mean(neg_log_p)

    def _binary_cross_entropy(self, X, mu_x, logstd_x):
        X_hat = mu_x + torch.randn_like(logstd_x) * torch.exp(logstd_x)
        return torch.nn.functional.binary_cross_entropy_with_logits(X_hat, X)
    
    def _kl_gaussian(self, mu, logstd):
        # DAG-GNN paper equation (8)
        return 0.5 * torch.mean((torch.sum(torch.exp(2*logstd) - 2*logstd + mu**2 - 1)))

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
