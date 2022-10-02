from typing import Optional, Any

import numpy as np

from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, TrainingBatchLoop

class GAE_torch_FitLoop(FitLoop):
    def __init__(self, min_epochs: int = 0, max_epochs: int = 1000) -> None:
        super().__init__(min_epochs, max_epochs)
        self.epoch_loop = GAE_torch_TrainingEpochLoop()
        
class GAE_torch_TrainingEpochLoop(TrainingEpochLoop):
    def __init__(self, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(min_steps, max_steps)
        self.batch_loop = GAE_torch_TrainingBatchLoop()

class GAE_torch_TrainingBatchLoop(TrainingBatchLoop):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = 0.0
        self.beta = 2.0
        self.gamma = 0.25
        self.rho = 1.0
        self.rho_thresh = 1e30
        self.h_thresh = 1e-8
        self.early_stopping_thresh = 1.0
        self.early_stopping = False

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self._alpha, self._beta, self._rho = self.alpha, self.beta, self.rho
        self._h, self._h_new = np.inf, np.inf
        self._perv_w_est, self._prev_mse = None, np.inf
        return super().on_advance_start(*args, **kwargs)
        
    def advance(self, batch: Any, batch_idx: int) -> None:
        while self._rho < self.rho_thresh:
            super().advance(batch, batch_idx)
            outputs = self._outputs.pop()
            
            self._h_new = outputs[0]['h']
            if self._h_new > self.gamma * self._h:
                self._rho *= self._beta
            else:
                break

        mse_new = outputs[0]['loss_mse']
        
        if self.early_stopping:
            if (mse_new / self._prev_mse > self.early_stopping_thresh
                and self._h_new <= 1e-7):
                self._outputs.append(outputs)
                return
            else:
                self._prev_w_est = outputs[0]['W']
                self._prev_mse = mse_new
        
        w_est, h = outputs[0]['W'], outputs[0]['h']
        self._alpha += self._rho * self._h_new

        if h <= self.h_thresh:
            self.skip()

        self._outputs.append(outputs)