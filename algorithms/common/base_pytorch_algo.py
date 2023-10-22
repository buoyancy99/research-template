from abc import ABC, abstractmethod
from typing import Any, Union
from lightning.pytorch.utilities.types import STEP_OUTPUT

from omegaconf import DictConfig
import lightning.pytorch as pl
import torch
import numpy as np
import wandb


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

    def log_video(self, video: Union[np.ndarray, torch.Tensor], name, fps=12, format="mp4"):
        """
        Log video to wandb.

        Args:
            video: a numpy array or tensor, either in form (time, channel, height, width) or in the form
                (batch, time, channel, height, width). The content must be be in 0-255 if under dtype uint8 
                or [0, 1] otherwise.
            name: the name of the video.
            fps: the frame rate of the video.
            format: the format of the video. Can be either "mp4" or "gif".
        """

        if isinstance(video, torch.Tensor):
            video = video.detech().cpu().numpy()
        
        if video.dtype != np.uint8:
            video = np.clip(video, a_min=0, a_max=1) * 255
            video = video.astype(np.uint8)

        self.logger.experiment.log({
            name: wandb.Video(video, fps=fps, format=format),
        }, step=self.global_step)
