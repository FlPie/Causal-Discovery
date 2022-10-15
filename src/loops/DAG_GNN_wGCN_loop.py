from inspect import Parameter
from typing import Optional, Any, T

import numpy as np

from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, TrainingBatchLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _set_sampler_epoch

class DAG_GNN_wGCN_FitLoop(FitLoop):
    def reset(self) -> None:
        self.current_iteration = 0
        self.outputs = []
        self.epoch_progress.reset()
        # self.trainer.reset_train_dataloader(self.trainer.lightning_module)
        return super().reset()

    def run(self, *args: Any, **kwargs: Any) -> T:
        for i in range(20):
            while self.trainer.model.c_A < 1e+20:
                # reset for training model again
                self.reset()
                # train model for specipic c & lambda
                # for epoch in range()
                output = super().run(*args, **kwargs)
                # self._debug_loop_done(i)
                
                # update c & lambda
                A_new = self.trainer.model.adj_A_new.data.clone()
                h_A_new = self.trainer.model._h_A(A_new, A_new.shape[0])
                if h_A_new.item() > 0.25 * self.trainer.model.h_A:
                    self.trainer.model.c_A *= 10
                else:
                    break
            self.trainer.model.h_A = h_A_new.item()
            self.trainer.model.lambda_A += self.trainer.model.c_A * h_A_new.item()
        return output

    def _debug_loop_done(self, i=None):
        print(f"########## {i}, {self.done}, {self.skip} ##########")
        print(f"{self.trainer.num_training_batches == 0}, "+
                f"{_is_max_limit_reached(self.epoch_loop.global_step, self.max_steps)}, "+ 
                f"{_is_max_limit_reached(self.epoch_progress.current.processed, self.max_epochs)}, "+
                f"{self.epoch_progress.current.processed}, {self.max_epochs}, "+
                f"{self.trainer.should_stop}, ")