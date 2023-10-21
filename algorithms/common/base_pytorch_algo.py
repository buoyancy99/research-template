from abc import ABC, abstractmethod
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT

from omegaconf import DictConfig
import lightning.pytorch as pl
import torch


class BasePytorchAlgo(pl.LightningModule, ABC):
    """
    A base class for Pytorch algorithms using Pytorch Lightning.
    See https://lightning.ai/docs/pytorch/stable/starter/introduction.html for more details.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.debug = self.cfg.debug
        self._build_model()

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)
    
    def configure_optimizers(self):
        parameters = self.parameters()
        return torch.optim.Adam(parameters, lr=self.cfg.lr)
