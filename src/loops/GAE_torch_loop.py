from typing import Optional, Any

import numpy as np

from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, TrainingBatchLoop
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature

class GAE_torch_FitLoop(FitLoop):
    def __init__(self, min_epochs: int = 0, max_epochs: int = 10, update_freq: int = 300) -> None:
        super().__init__(min_epochs, max_epochs)
        self.update_freq = update_freq
        self.alpha = 0.0
        self.beta = 2.0
        self.gamma = 0.25
        self.rho = 1.0
        self.rho_thresh = 1e30
        self.h_thresh = 1e-8
        self.early_stopping_thresh = 1.0
        self.early_stopping = False
        self.h, self.h_new = np.inf, np.inf
        self.perv_w_est, self.prev_mse = None, np.inf
        
    def advance(self) -> None:
        while self.rho < self.rho_thresh:
            for _ in range(self.update_freq):
                super().advance()
            outputs = self._outputs[-1][0][0]
            
            self.h_new = outputs['h']
            if self.h_new > self.gamma * self.h:
                self.rho *= self.beta
            else:
                break

        mse_new = outputs['loss_mse']
        
        if self.early_stopping:
            if (mse_new / self.prev_mse > self.early_stopping_thresh
                and self.h_new <= 1e-7):
                return
            else:
                self.prev_w_est = outputs['W']
                self.prev_mse = mse_new
        
        w_est, h = outputs['W'], outputs['h']
        self.alpha += self.rho * self.h_new

        if self.h <= self.h_thresh:
            self.skip()