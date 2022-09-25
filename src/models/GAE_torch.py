# from https://github.com/ThinkNaive/GAE-PSP
# in progress
import os
from pickletools import optimize
from turtle import forward
from tqdm import tqdm
import math
from typing import Any

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import wandb
import hydra

import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl

import torch_geometric as pyg
import torch_geometric.nn as geo_nn


class BatchNorm(nn.Module):
    def __init__(
        self,
        num_features,
        permute=(0, 2, 1),
        eps=0.00001,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self._p = permute
        self._bn = nn.BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, x):
        x = self._bn(x.permute(*self._p)).permute(*self._p)
        return x


class NonLinerTransformer(nn.Module):
    def __init__(self, in_dim, n_hid, out_dim, n_layer):
        """Component for encoder and decoder
        Args:
            in_dim (int): input dimension.
            n_hid (int): model layer dimension.
            out_dim (int): output dimension.
        """
        super(NonLinerTransformer, self).__init__()
        dims = (
            [(in_dim, n_hid)]
            + [(n_hid, n_hid) for _ in range(n_layer - 1)]
            + [(n_hid, out_dim)]
        )
        fc_layers = [nn.Linear(pair[0], pair[1]) for pair in dims]
        # bn_layers = [BatchNorm(n_hid) for _ in range(n_layer)]
        lr_layers = [nn.LeakyReLU(0.05) for _ in range(n_layer)]
        layers = []
        for i in range(n_layer):
            layers.append(fc_layers[i])
            # layers.append(bn_layers[i])
            layers.append(lr_layers[i])
        layers.append(fc_layers[-1])
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


class CAE(pl.LightningModule):
    def __init__(
        self, 
        d=11, 
        n_dim=1, 
        n_hid=4, 
        n_latent=4, 
        n_layer=3, 
        lambda_sparsity=1.0, 
        psp=False,
        *args, 
        **kwargs):
        """
        Graph Autoencoder
        Code modified from:
            https://github.com/huawei-noah/trustworthyAI/blob/master/Causal_Structure_Learning/GAE_Causal_Structure_Learning/src/models/gae.py
        Args:
            d (int): variables number
            n_dim (int): features number, aka input dimension.
            n_hid (int): encoder and decoder layer dimension.
            n_latent (int): encoded latent layer dimension.
        """
        super(CAE, self).__init__()

        self.save_hyperparameters(logger=False)

        # constant
        self._d = d
        self._l1 = lambda_sparsity
        self._psp = psp
        # ground truth adj matric
        gt_df = pd.read_csv(self.hparams.gt_path, index_col=0)
        self.ground_truth_G = nx.from_pandas_adjacency(gt_df, create_using=nx.DiGraph)

        # hparams
        self.alpha = 0.0
        self.rho = 1.0
        self.graph_thresh = 0.3

        # Non-linear transformer for data dimensions
        self.encoder = NonLinerTransformer(n_dim, n_hid, n_latent, n_layer)
        self.decoder = NonLinerTransformer(n_latent, n_hid, n_dim, n_layer)

        # Test: Non-linear invertible transformer for data dimensions
        # self.encoder = InvTransformer(n_dim, n_hid, n_latent, n_layer)
        # self.decoder = InvTransformer(n_latent, n_hid, n_dim, n_layer)

        # initial value of W has substantial impact on model performance
        _mask = torch.Tensor(1 - np.eye(d))
        self.register_buffer("_mask", _mask)
        self._W_pre = nn.Parameter(
            torch.Tensor(np.random.uniform(low=-0.1, high=0.1, size=(d, d)))
        )

        self.optimizer = self.hparams.optimizer(params=self.parameters())

    def forward(self, x):
        # constant
        x=x.reshape(512,11,1)
        n = x.shape[0]
        # get weighted matrix parameter
        # https://github.com/huawei-noah/trustworthyAI/issues/21#event-4920062931
        self.W = self._W_pre * self._mask

        # model forward
        self.h1 = self.encoder(x)
        self.h2 = torch.einsum("ijk,jl->ilk", self.h1, self.W)
        self.x_hat = self.decoder(self.h2)

        # compute loss
        loss_mse = torch.square(torch.norm(self.x_hat - x))
        loss_sparsity = self._compute_l1(self.W)
        h = torch.trace(torch.matrix_exp(self.W * self.W)) - self._d
        loss = (
            0.5 / n * loss_mse
            + self._l1 * loss_sparsity
            + self.alpha * h
            + 0.5 * self.rho * h * h
        )

        # for debug and watch
        # mask = torch.Tensor(list((0, 1) + (0,) * 98)).cuda()
        # xmsk = x_hat.squeeze().mean(dim=0)
        # xhc = (x_hat.squeeze() * (1 - mask) + xmsk * mask).unsqueeze(-1)
        # lmc = 0.5 / n * torch.square(torch.norm(xhc - x))

        return loss, loss_mse, loss_sparsity, h, self.W

    def training_step(self, batch):
        loss, loss_mse, loss_sparsity, h, W = self.forward(batch.x)
        return {
            'loss': loss,
            'loss_mse': loss_mse,
            'loss_sparsity': loss_sparsity,
            'h': h,
            'W': W,
        }

    def training_epoch_end(self, outputs) -> None:
        final = outputs[-1]
        self.log("train/losses", final, on_step=False, on_epoch=True, prog_bar=False)

        W = final['W']
        W = W.cpu().detach().cpu().numpy()
        W = W / np.max(np.abs(W))
        W[abs(W) < self.graph_thresh] = 0
        causal_matrix = W
        graph = nx.DiGraph(causal_matrix)

        accu = self._count_accuracy(self.ground_truth_G, graph)

        self.log("train/accu", accu, on_step=False, on_epoch=True, prog_bar=False)
        return None

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer
        }

    def _compute_l1(self, W):
        if self._psp:
            W = W / torch.max(torch.abs(W))
            loss = 2 * torch.sigmoid(2 * W) - 1
            return loss.norm(p=1) / self._d ** 2
        else:
            return W.norm(p=1)

    def _count_accuracy(
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