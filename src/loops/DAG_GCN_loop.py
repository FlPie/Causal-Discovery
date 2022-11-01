from typing import Optional, Any, T

import numpy as np

from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, TrainingBatchLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _set_sampler_epoch

import wandb
import networkx as nx
from ..models.components import DAG_utils as DAG_utils

class DAG_GCN_FitLoop(FitLoop):
    def __init__(self, min_epochs: int = 0, max_epochs: int = 1000, k_max_iter: int=10) -> None:
        super().__init__(min_epochs, max_epochs)
        self.k_max_iter = k_max_iter

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
            self._log_bests()
            if self.trainer.model.h_A <= 1e-8:
                break
            
        return output

    def _log_bests(self):
        gt_G = self.trainer.model.ground_truth_G
        # best shd
        G = self.trainer.model.best_shd_G
        graph = nx.to_numpy_array(G)
        graph_img, adj_img = DAG_utils.get_plot_imgs(graph, G, gt_G)
        wandb.log({"best/shd_graph": wandb.Image(graph_img)})
        wandb.log({"best/shd_adj": wandb.Image(adj_img)})
        wandb.log({"best/shd":self.trainer.model.best_shd})
        # best tpr
        G = self.trainer.model.best_tpr_G
        graph = nx.to_numpy_array(G)
        graph_img, adj_img = DAG_utils.get_plot_imgs(graph, G, gt_G)
        wandb.log({"best/tpr_graph": wandb.Image(graph_img)})
        wandb.log({"best/tpr_adj": wandb.Image(adj_img)})
        wandb.log({"best/tpr":self.trainer.model.best_tpr})
        # best loss
        G = self.trainer.model.best_loss_G
        graph = nx.to_numpy_array(G)
        graph_img, adj_img = DAG_utils.get_plot_imgs(graph, G, gt_G)
        wandb.log({"best/loss_graph": wandb.Image(graph_img)})
        wandb.log({"best/loss_adj": wandb.Image(adj_img)})
        wandb.log({"best/loss":self.trainer.model.best_loss})
        

    def _debug_loop_done(self, i=None):
        print(f"########## {i}, {self.done}, {self.skip} ##########")
        print(f"{self.trainer.num_training_batches == 0}, "+
                f"{_is_max_limit_reached(self.epoch_loop.global_step, self.max_steps)}, "+ 
                f"{_is_max_limit_reached(self.epoch_progress.current.processed, self.max_epochs)}, "+
                f"{self.epoch_progress.current.processed}, {self.max_epochs}, "+
                f"{self.trainer.should_stop}, ")