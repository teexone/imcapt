from collections import defaultdict
from pytorch_lightning.loggers import Logger
from typing import Dict, Optional

import torch 
import os
import datetime
import logging

class ModelLogger:

    def __init__(self) -> None:
        self.buff = defaultdict(list)

        self.logger = logging.getLogger('log')
        self.logger.setLevel(level=logging.INFO)

        os.makedirs("log", exist_ok=True)
        dt = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.path = f"logs/{dt}"
        os.makedirs(self.path, exist_ok=True)
        
        self.handler = None
        self.ind = 0
    

    def reHandle(self, filename: str):
        if self.logger.hasHandlers():
            self.logger.removeHandler(self.handler)
        self.handler = logging.FileHandler(filename, mode="a+")
        self.handler.setLevel(level=logging.INFO)
        self.logger.addHandler(self.handler)

    def save(self) -> None:
        for fold, vals in self.buff.items():
            os.makedirs(f"{self.path}/{fold}", exist_ok=True)
            for (epoxide, step, v) in vals:     
                if isinstance(v, list):
                    if len(v) > 0 and isinstance(v[0], list):
                        v = "\n".join([" ".join(x) for x in v])
                    else:
                        v = " ".join(v)           
                step = step if step is not None else "metrics"
                epoxide = epoxide if epoxide is not None else ""
                self.reHandle(f"{self.path}/{fold}/{epoxide}_{step}.txt")
                self.logger.info(str(v))
        self.buff.clear()
    
    def log(self, 
            metrics: Dict[str, float|torch.Tensor], 
            step: Optional[int] = None, 
            epoch=None) -> None:
        
        for name, value in metrics.items():
            name = name.replace('_', '/')
            self.buff[name].append((epoch, step, value,))
