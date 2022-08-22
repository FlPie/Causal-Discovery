import os
from tqdm import tqdm
import math

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl

import torch_geometric as pyg


MAX_LOGSTD=10


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, adj_A):
        """"""
        super(GNNEncoder, self).__init__()

        self.adj_A = nn.Parameter(torch.autograd.Variable(
            adj_A, requires_grad=True
        ))
        self.Wa = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        
        self.mlp_shared = pyg.nn.Sequential('x', [
            (pyg.nn.Linear(in_channels, hidden_channels), 'x -> x'),
            nn.ReLU(inplace=True),
        ])
        
        self.mlp_mu = pyg.nn.Sequential('x', [
            (pyg.nn.Linear(hidden_channels, out_channels), 'x -> x'),
        ])
        
        self.mlp_logvar = pyg.nn.Sequential('x', [
            (pyg.nn.Linear(hidden_channels, out_channels), 'x -> x'),
        ])

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
    def forward(self, x):
        x = self.mlp_shared(x)

        adj_A1 = torch.sinh(3.0 * self.adj_A).type_as(x)
        
        adj = self.adj_process(adj_A1)
        
        mu = self.mlp_mu(x)
        mu = torch.matmul(adj, mu + self.Wa) - self.Wa
        
        logvar = self.mlp_logvar(x)
        logvar = torch.matmul(adj, logvar)
        
        return mu, logvar, adj_A1, self.Wa
            
        
    def adj_process(self, adj):
        '''
        return I-A^T
        '''
        return torch.eye(adj.shape[0]).float().type_as(adj) - adj.transpose(0,1)


class GNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNDecoder, self).__init__()
        
        self.mlp_shared = pyg.nn.Sequential('x', [
            (pyg.nn.Linear(in_channels, hidden_channels), 'x -> x'),
            nn.ReLU(inplace=True),
        ])
        
        self.mlp_mu = pyg.nn.Sequential('x', [
            (pyg.nn.Linear(hidden_channels, out_channels), 'x -> x'),
        ])
        
        self.mlp_logvar = pyg.nn.Sequential('x', [
            (pyg.nn.Linear(hidden_channels, out_channels), 'x -> x'),
        ])

        self.init_weights()

        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, z, adj, Wa):
        z_ = z.view(-1, 1)
        adj_hat = torch.sigmoid(torch.matmul(z_, z_.transpose(0,1)))

        adj = self.adj_process_inverse(adj)
        
        x = torch.matmul(adj, z + Wa) - Wa
        x = self.mlp_shared(x)

        mu = self.mlp_mu(x)
        
        logvar = self.mlp_mu(x)
        
        return mu, logvar, adj_hat

        
    def adj_process_inverse(self, adj):
        '''
        return (I - A^T)^-1
        '''
        return torch.inverse(
            torch.eye(adj.shape[0]).float().type_as(adj) - adj.transpose(0,1))


class DAG_GNN(pl.LightningModule):
    def __init__(
        self, 
        optimizer,
        lambda_A=0,
        c_A=1,
        threshold=0.3,
        adj_A = None,
        **kwargs):
        super(DAG_GNN, self).__init__()
        
        self.save_hyperparameters(logger=False)

        gt_df = pd.read_csv(self.hparams.gt_path, index_col=0)
        self.ground_truth_G = nx.from_pandas_adjacency(gt_df, create_using=nx.DiGraph)
        
        # self.adj_A = torch.zeros(
        #     [self.hparams.num_features,self.hparams.num_features], 
        #     dtype=torch.float)
        gt_np = nx.to_numpy_array(self.ground_truth_G)
        gt_np = gt_np * 0.3
        gt_np[2:,2:] = 0
        self.adj_A = torch.tensor(gt_np)

        self.encoder = GNNEncoder(
                in_channels=self.hparams.num_features * self.hparams.batch_size,
                hidden_channels=self.hparams.hidden_size,
                out_channels=self.hparams.num_features,
                adj_A = self.adj_A)
        self.decoder = GNNDecoder(
                in_channels=self.hparams.num_features,
                hidden_channels=self.hparams.num_features,
                out_channels=self.hparams.num_features * self.hparams.batch_size)

        self.lambda_A = lambda_A
        self.c_A = c_A
        self.h_A = np.inf
        self.threshold = threshold

        self.optimizer = self.hparams.optimizer(params=self.parameters())
        
        
    def forward(self, x, lambda_A, c_A):
        z, adj_A1, Wa = self.encode(x)
        self.x_hat, adj_hat = self.decode(z, adj_A1, Wa)
        losses = self.loss(x, adj_A1, lambda_A, c_A)
        return self.x_hat, losses, adj_A1
        
        
    def step(self, batch):
        x_hat, loss, self.adj_A_new = self.forward(batch.x, self.lambda_A, self.c_A)
        return x_hat, loss
        
    
    def training_step(self, batch):
        x_hat, losses = self.step(batch)
        return {
                'loss'      : losses['loss'],
                'nll_loss'  : losses['nll_loss'],
                'kl_loss'   : losses['kl_loss'],
                'elbo_loss' : losses['elbo_loss'],
                'A_loss'    : losses['A_loss']}
    
    
    def training_epoch_end(self, outputs):
        loss      = np.sum(d['loss'] for d in outputs) / (len(outputs)) # * self.hparams.batch_size)
        nll_loss  = np.sum(d['nll_loss'] for d in outputs) / (len(outputs)) # * self.hparams.batch_size)
        kl_loss   = np.sum(d['kl_loss'] for d in outputs) / (len(outputs)) # * self.hparams.batch_size)
        elbo_loss = np.sum(d['elbo_loss'] for d in outputs) / (len(outputs)) # * self.hparams.batch_size)
        A_loss    = np.sum(d['A_loss'] for d in outputs) / (len(outputs)) # * self.hparams.batch_size)
        losses = {
                'loss'      : loss,
                'nll_loss'  : nll_loss,
                'kl_loss'   : kl_loss,
                'elbo_loss' : elbo_loss,
                'A_loss'    : A_loss,
        }
        adj_hat = self.adj_A_new
        graph = adj_hat.cpu().clone().detach().numpy()
        # graph[np.abs(graph) < self.threshold] = 0
        graph[graph < self.threshold] = 0
        accu = self.count_accuracy(self.ground_truth_G, nx.DiGraph(graph))
        self.log("train/losses", losses, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/accu", accu, on_step=False, on_epoch=True, prog_bar=False)
        plt.imshow(graph)
        plt.colorbar()
        wandb.log({"train/adj_plot": plt})
        plt.clf()

        # update parameters
        A_new = adj_hat.clone()
        h_A_new = self._h_A(A_new, self.hparams.num_features)
        if h_A_new.item() > 0.25 * self.h_A:
            self.c_A *= 10
        else:
            pass


    def validation_epoch_end(self):
        A_new = self.adj_A_new.clone()
        m = A_new.shape[0]
        h_A_new = self._h_A(A_new, m)
        self.h_A = h_A_new.item()
        self.lambda_A += self.c_A * h_A_new.item()
        self.update_optimizer()


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.optimizer,
        }


    def update_optimizer(self):
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        for parame_group in self.optimizer.param_groups:
            original_lr = parame_group["lr"]

        estimated_lr = original_lr / (math.log10(self.c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in self.optimizer.param_groups:
            parame_group["lr"] = lr

        #return self.optimizer

    
    def count_accuracy(
        self,
        G_true: nx.DiGraph,
        G: nx.DiGraph,
        G_und: nx.DiGraph = None) -> tuple:
        """Compute FDR, TPR, and FPR for B, or optionally for CPDAG B + B_und.
        Args:
            G_true: ground truth graph
            G: predicted graph
            G_und: predicted undirected edges in CPDAG, asymmetric
        Returns:
            fdr: (reverse + false positive) / prediction positive
            tpr: (true positive) / condition positive
            fpr: (reverse + false positive) / condition negative
            shd: undirected extra + undirected missing + reverse
            nnz: prediction positive
        """
        B_true = nx.to_numpy_array(G_true) != 0
        B = nx.to_numpy_array(G) != 0
        B_und = None if G_und is None else nx.to_numpy_array(G_und)
        d = B.shape[0]
        # linear index of nonzeros
        if B_und is not None:
            pred_und = np.flatnonzero(B_und)
        pred = np.flatnonzero(B)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        if B_und is not None:
            # treat undirected edge favorably
            true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
            true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        if B_und is not None:
            false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
            false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred)
        if B_und is not None:
            pred_size += len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        B_lower = np.tril(B + B.T)
        if B_und is not None:
            B_lower += np.tril(B_und + B_und.T)
        pred_lower = np.flatnonzero(B_lower)
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        return {'fdr':fdr, 'tpr':tpr, 'fpr':fpr, 'shd':shd, 'pred_size':pred_size}

    
    def loss(self, x, adj, lambda_A, c_A):
        # nll_gaussian
        nll_loss = self.nll_gaussian(x)
        
        # kl_gaussian_sem
        # kl_loss = self.kl_loss()
        kl_loss = self.kl_loss_sem()
        
        # DAG-GNN paper (4)
        # L^k_ELBO ≡ −D_KL(q(Z|Xk) || p(Z)) + E_q(Z|X^k)[log p(X^k|Z)]
        elbo_loss = nll_loss + kl_loss
        
        # h(A)
        m = adj.shape[0]
        h_A = self._h_A(adj, m)
        sparse_loss = self.sparse_loss(adj)
        
        # lambda * h(A) + 0.5 * c |h(A)|^2
        A_loss = self.A_loss(h_A, lambda_A, c_A, sparse_loss)
        
        loss = elbo_loss + A_loss

        losses = {'loss': loss, 
        'nll_loss': nll_loss, 
        'kl_loss': kl_loss, 
        'elbo_loss': elbo_loss, 
        'h_A': h_A, 
        'A_loss': A_loss}
        
        return losses
    
    
    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__, adj_A1, Wa = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        return self.__mu__, adj_A1, Wa
    
    
    def decode(self, *args, **kwargs):
        """"""
        self.__x_mu__, self.__x_logstd__, self.__adj_hat__ = self.decoder(*args, **kwargs)
        self.__x_logstd__ = self.__x_logstd__.clamp(max=MAX_LOGSTD)
        return self.__x_mu__, self.__adj_hat__
    
    def A_loss(self, h_A, lambda_A, c_A, sparse_loss):
        # lambda * h(A) + 0.5 * c |h(A)|^2
        # what's "+ (100.0 * torch.trace(adj * adj)) + sparse_loss" for ??
        # lambda_A : hparam
        # c_A      : hparam
        A_loss = (lambda_A * h_A) + (0.5 * c_A * h_A * h_A) + sparse_loss
        return A_loss
    
    
    def _h_A(self, adj, m):
        # DAG-GNN paper (13)
        # h(A) = tr[(I + alpha * A * A)^m] - m
        matrix_poly = torch.eye(m).float().type_as(adj) + torch.div(adj*adj, m)
        expm_A = torch.matrix_power(matrix_poly, m)
        h_A = torch.trace(expm_A) - m
        return h_A


    def sparse_loss(self, adj, tau=0):
        return (100.0 * torch.trace(adj * adj)) + tau * torch.sum(torch.abs(adj))
    
    
    def nll_gaussian(self, x, x_hat=None, variance=0, add_const=False):
        """"""
        # DAG-GNN paper (9)
        # E_q(Z|X)[log p(X|Z)] ≈ 
        # 1/L ∑^L_l=1 ∑^m_i=1 ∑^d_j=1 −((X_ij − (M^(l)_X)_ij )^2 / 2(S^(l)_X)^2_ij) − log(S^(l)_X)_ij − c
        x = x
        x_hat = self.__x_mu__ if x_hat is None else x_hat
        variance = variance
        
        neg_log_p = variance + torch.div(torch.pow(x_hat - x, 2), 2.0 * np.exp(2.0 * variance))
        
        if add_const:
            const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
            neg_log_p += const
        
        return torch.mean(neg_log_p)
    
    
    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        # DAG-GNN paper (8)
        # −D_KL(q(Z|Xk) || p(Z)) = 1/2 ∑^m_i=1 ∑^d_j=1 (S_Z)^2_ij + (M_Z)^2_ij − 2log(S_Z)_ij − 1
        # return -0.5 * torch.mean(
        #     torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=0))
        return -0.5 * (
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=0) / mu.size(0))


    def kl_loss_sem(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        # DAG-GNN paper (8)
        # −D_KL(q(Z|Xk) || p(Z)) = 1/2 ∑^m_i=1 ∑^d_j=1 (S_Z)^2_ij + (M_Z)^2_ij − 2log(S_Z)_ij − 1
        return 0.5 * (torch.sum(mu**2, dim=0) / mu.size(0))