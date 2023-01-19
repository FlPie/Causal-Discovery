from typing import Optional, Any, T

import numpy as np
import pandas as pd
import xarray as xr
import json

from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, TrainingBatchLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _set_sampler_epoch

import wandb
import networkx as nx
from ..models.components import DAG_utils as DAG_utils

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()

class DAG_GCN_FitLoop(FitLoop):
    def __init__(self, min_epochs: int = 0, max_epochs: int = 1000, k_max_iter: int=10) -> None:
        super().__init__(min_epochs, max_epochs)
        self.k_max_iter = k_max_iter
        self.shd_G = None
        self.tpr_G = None
        self.loss_G = None
        self.last_G = None
        self.dims=["G_type", "iter", "E_type", "node", "in_degree", "out_degree"]
        self.xr_list = []
        # self.xarray = xr.Dataset(dims=self.dims)

    def reset(self) -> None:
        self.current_iteration = 0
        self.outputs = []
        self.epoch_progress.reset()
        # self.trainer.reset_train_dataloader(self.trainer.lightning_module)
        return super().reset()

    def run(self, *args: Any, **kwargs: Any) -> T:
        for i in range(self.k_max_iter):
            while self.trainer.model.c_A < 1e+20:
                # reset for training model again
                self.reset()
                # train model for specipic c & lambda
                output = super().run(*args, **kwargs)
                # self._debug_loop_done(i)
                
                # update c & lambda
                A_new = self.trainer.model.dense_adj.data.clone()
                h_A_new = self.trainer.model._h_A(A_new)
                if h_A_new.item() > 0.25 * self.trainer.model.h_A:
                    self.trainer.model.c_A *= 10
                else:
                    break
            self.trainer.model.h_A = h_A_new.item()
            self.trainer.model.lambda_A += self.trainer.model.c_A * h_A_new.item()
            # log best adjs
            self._log_bests(i)
            if self.trainer.model.h_A <= 1e-8:
                break

        # self.xr = xr.concat(self.xr_list, dim="G_type")
        # self.xr.transpose("G_type", "iter", "E_type", "node", "in_degree", "out_degree")
        # self.df = self.xr.to_dataframe()
        with open("../../train_epoch300_iter100_2.json", "w") as f:
            json.dump(self.xr_list, f)
        # wandb.log({"table": self.df})
            
        return output

    def _log_bests(self, iter):
        gt_G = self.trainer.model.ground_truth_G
        if None in (self.shd_G, self.tpr_G, self.loss_G, self.last_G): 
            self.shd_G = gt_G
            self.tpr_G = gt_G
            self.loss_G = gt_G
            self.last_G = gt_G
            self._compare_graphs(gt_G, gt_G, gt_G, "gt", iter)
        # best shd
        G = self.trainer.model.best_shd_G
        self._log_graph(gt_G, G, "shd")
        self._compare_graphs(gt_G, G, self.shd_G, "shd", iter)
        self.shd_G = G
        # best tpr
        G = self.trainer.model.best_tpr_G
        self._log_graph(gt_G, G, "tpr")
        self._compare_graphs(gt_G, G, self.tpr_G, "tpr", iter)
        self.tpr_G = G
        # best loss
        G = self.trainer.model.best_loss_G
        self._log_graph(gt_G, G, "loss")
        self._compare_graphs(gt_G, G, self.loss_G, "loss", iter)
        self.loss_G = G
        # last G
        G = self.trainer.model.last_G
        self._log_graph(gt_G, G, "last")
        self._compare_graphs(gt_G, G, self.last_G, "last", iter)
        self.last_G = G

    def _log_graph(self, gt_G, G, name):
        graph = nx.to_numpy_array(G)
        graph_img, adj_img = DAG_utils.get_plot_imgs(graph, G, gt_G)
        wandb.log({"best/"+name+"_graph": wandb.Image(graph_img)})
        wandb.log({"best/"+name+"_adj": wandb.Image(adj_img)})
        wandb.log({"best/"+name:self.trainer.model.best_shd})
        
    def _compare_graphs(self, gt_G, G, last_G, name, iter):
        # calculate shd & tpr
        accu = DAG_utils.count_accuracy(gt_G, G)

        I = nx.intersection(gt_G, G) # intersection, correct
        R = nx.intersection(gt_G, nx.reverse(G)) # reverse intersection, reversed
        W = nx.difference(G, gt_G)

        node_lists = list(G.nodes)

        # in degrees
        G_in = list(G.in_degree(node_lists))
        I_in = list(I.in_degree(node_lists))
        R_in = list(R.in_degree(node_lists))
        W_in = list(W.in_degree(node_lists))

        # out degrees
        G_out = list(G.out_degree(node_lists))
        I_out = list(I.out_degree(node_lists))
        R_out = list(R.out_degree(node_lists))
        W_out = list(W.out_degree(node_lists))

        # debug log
        print(f"iter {iter} {name} shd: {accu['shd']}, tpr: {accu['tpr']}")
        n = "node"
        a = "G"
        b = "C"
        c = "W"
        print("----------------------------------------")
        print("IN DEGREES")
        print("----------------------------------------")
        print(f"{n:>10}: {a:>4} {b:>4} {c:>4}")
        for g, i, w in zip(G_in, I_in, W_in):
            color = Fore.GREEN
            if i[1]+w[1] != g[1]:
                color = Fore.RED
            print(f"{color}{g[0]:>10}: {g[1]:>4} {i[1]:>4} {w[1]:>4}{Style.RESET_ALL}")
        print("----------------------------------------")
        print("OUT DEGREES")
        print("----------------------------------------")
        print(f"{n:>10}: {a:>4} {b:>4} {c:>4}")
        for g, i, w in zip(G_out, I_out, W_out):
            color = Fore.GREEN
            if i[1]+w[1] != g[1]:
                color = Fore.RED
            print(f"{color}{g[0]:>10}: {g[1]:>4} {i[1]:>4} {w[1]:>4}{Style.RESET_ALL}")

        # log results
        # Graph
        for i, o in zip(G_in, G_out):
            if i[0] == o[0]:
                self._append_xarray(name, iter, "Graph", i[0], i[1], o[1])
            else:
                raise ValueError("node order not match")

        # Correct
        for i, o in zip(I_in, I_out):
            if i[0] == o[0]:
                self._append_xarray(name, iter, "Correct", i[0], i[1], o[1])
            else:
                raise ValueError("node order not match")

        # Reversed
        for i, o in zip(R_in, R_out):
            if i[0] == o[0]:
                self._append_xarray(name, iter, "Reversed", i[0], i[1], o[1])
            else:
                raise ValueError("node order not match")

        # Wrong
        for i, o in zip(W_in, W_out):
            if i[0] == o[0]:
                self._append_xarray(name, iter, "Wrong", i[0], i[1], o[1])
            else:
                raise ValueError("node order not match")

    def _append_xarray(self, name, iter, E_type, node, in_dgr, out_dgr):
        G_xr = {
            "G_type": name,
            "iter": iter,
            "E_type": E_type,
            "node": node,
            "in_degree": in_dgr,
            "out_degree": out_dgr,
        }
        self.xr_list.append(G_xr)

    def _debug_loop_done(self, i=None):
        print(f"########## {i}, {self.done}, {self.skip} ##########")
        print(f"{self.trainer.num_training_batches == 0}, "+
                f"{_is_max_limit_reached(self.epoch_loop.global_step, self.max_steps)}, "+ 
                f"{_is_max_limit_reached(self.epoch_progress.current.processed, self.max_epochs)}, "+
                f"{self.epoch_progress.current.processed}, {self.max_epochs}, "+
                f"{self.trainer.should_stop}, ")